#Test semi_supervised.py
from pytorch_lightning import Trainer
from src import semi_supervised, predict
from src.models import joint_semi
from src.models.multi_stage import TreeDataset
import pytest
import numpy as np

@pytest.fixture()
def prediction_model_path(m, tmpdir):    
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    trainer.fit(m)
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
    train = semi_supervised.create_dataframe(config, unlabeled_df=dm.train)
    train.shape[0] == config["semi_supervised"]["num_samples"]
    assert all(train.index.values == np.arange(train.shape[0]))

def test_fit(config, dm, prediction_model_path, ROOT, prediction_dir, comet_logger):
    """Test that the model can load an existing model for unlabeled predictions and train """
    config["semi_supervised"]["site_filter"] = None
    config["semi_supervised"]["crop_dir"] = prediction_dir
    config["semi_supervised"]["threshold"] = 0
    config["semi_supervised"]["model_path"] = prediction_model_path    
    
    taxonomic_csv = "{}/data/raw/families.csv".format(ROOT)            
    
    joint_model = joint_semi.TreeModel(
        config=config,
        supervised_test=dm.test,
        supervised_train=dm.train,
        taxonomic_csv=taxonomic_csv
    )
    
    trainer = Trainer(
        fast_dev_run=False,
        max_epochs=1,
        limit_train_batches=1,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        multiple_trainloader_mode="min_size",
        logger=comet_logger
    )
    
    trainer.fit(joint_model)
    ds = TreeDataset(df=dm.test, train=False, config=config)
    predictions = trainer.predict(joint_model, dataloaders=joint_model.predict_dataloader(ds))
    results = joint_model.gather_predictions(predictions)    
    
    assert not results.empty