#Test semi_supervised.py
from pytorch_lightning import Trainer
from src import semi_supervised, predict
from src.models import Hang2020, multi_stage, joint_semi
import pytest
import numpy as np

@pytest.fixture()
def prediction_dir(ROOT, config, m, tmpdir):
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    config["prediction_crop_dir"] = tmpdir  
    
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif".format(ROOT)    
    crowns = predict.find_crowns(rgb_path, config)    
    crown_annotations_path = predict.generate_prediction_crops(crowns=crowns, config=config)
    
    trees = predict.predict_tile(
        crown_annotations=crown_annotations_path,
        m=m,
        trainer=trainer,
        filter_dead=True,
        savedir=tmpdir,
        config=config)
    
    return prediction_dir

def test_create_dataframe(config, m, dm):
    config["semi_supervised"]["crop_dir"] = config["crop_dir"]
    config["semi_supervised"]["threshold"] = 0
    config["semi_supervised"]["num_samples"] = 5
    
    dm.train = semi_supervised.create_dataframe(config, m=m, unlabeled_df=dm.train, label_to_taxon_id=dm.label_to_taxonID)
    dm.train.shape[0] == config["semi_supervised"]["num_samples"]
    assert all(dm.train.score > config["semi_supervised"]["threshold"])
    assert all(dm.train.index.values == np.arange(dm.train.shape[0]))
    
def test_fit(config, dm, prediction_dir):
    #predict a few 
    config["semi_supervised"]["crop_dir"] = prediction_dir
    model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=dm.num_classes)        
    joint_model = joint_semi(
        model=model, 
        config=config,
        classes=dm.num_classes, 
        loss_weight=None,
        label_dict=dm.species_label_dict)
    
    trainer = Trainer(fast_dev_run=False, max_epochs=1, limit_train_batches=1, enable_checkpointing=False, num_sanity_val_steps=0)
    trainer.fit(joint_model)
    results = joint_model.evaluate_crowns(
        dm.val_dataloader(),
        crowns = dm.crowns,
    )    