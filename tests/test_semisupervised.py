#Test semi_supervised.py
from pytorch_lightning import Trainer
from src import semi_supervised, predict
from src.models import Hang2020, baseline, joint_semi
import pytest
import numpy as np
    

@pytest.fixture()
def prediction_model_path(dm, config, tmpdir):
    model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=dm.num_classes) 
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)    
    model_for_unlabeled_prediction = baseline.TreeModel(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=None,
        label_dict=dm.species_label_dict)
    
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    trainer.fit(model_for_unlabeled_prediction, datamodule=dm)
    trainer.save_checkpoint("{}/checkpoint.pt".format(tmpdir))
    
    return "{}/checkpoint.pt".format(tmpdir)

@pytest.fixture()
def prediction_dir(ROOT, config, tmpdir):
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    config["prediction_crop_dir"] = tmpdir  
    
    rgb_paths = ["{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif".format(ROOT),
                 "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2018.tif".format(ROOT)]  
    
    for rgb_path in rgb_paths:
        crowns = predict.find_crowns(rgb_path, config)    
        crown_annotations_path = predict.generate_prediction_crops(crowns=crowns, config=config)
    
    return tmpdir

def test_create_dataframe(config, prediction_model_path, dm):
    config["semi_supervised"]["crop_dir"] = config["crop_dir"]  
    config["semi_supervised"]["model_path"] = prediction_model_path    
    train = semi_supervised.create_dataframe(config, unlabeled_df=dm.train, label_to_taxon_id=dm.label_to_taxonID)
    train.shape[0] == config["semi_supervised"]["num_samples"]
    assert all(train.index.values == np.arange(train.shape[0]))

def test_fit(config, dm, prediction_model_path, prediction_dir):
    """Test that the model can load an existing model for unlabeled predictions and train """
    config["semi_supervised"]["site_filter"] = None
    config["semi_supervised"]["crop_dir"] = prediction_dir
    config["semi_supervised"]["threshold"] = 0
    config["semi_supervised"]["model_path"] = prediction_model_path    
    
    model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=dm.num_classes)        
    joint_model = joint_semi.TreeModel(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=None,
        supervised_test=dm.test,
        supervised_train=dm.train,
        label_dict=dm.species_label_dict)
    
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0, multiple_trainloader_mode="min_size")
    trainer.fit(joint_model)
    results = joint_model.evaluate_crowns(
        dm.val_dataloader(),
        crowns = dm.crowns
    )
    
    assert not results.empty