import os
import glob
import pytest
from src import predict
from src.models import multi_stage
from pytorch_lightning import Trainer
import cProfile, pstats
import pandas as pd

#Training module
@pytest.fixture()
def species_model_path(config, dm, ROOT, tmpdir):
    config["batch_size"] = 16    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)    
    m.ROOT = "{}/tests/".format(ROOT)
    trainer = Trainer(fast_dev_run=False, max_steps=1, limit_val_batches=1)
    trainer.fit(m)
    trainer.save_checkpoint("{}/model.pl".format(tmpdir))
    
    return "{}/model.pl".format(tmpdir)
    
def test_predict_tile(species_model_path, config, ROOT, tmpdir):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2018.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    hsi_pool = glob.glob(config["HSI_sensor_pool"])
    
    crowns = predict.find_crowns(rgb_path, config)
    trees = predict.predict_tile(
        crowns=crowns,
        img_pool=hsi_pool,
        filter_dead=False,
        species_model_path=species_model_path,
        savedir=tmpdir,
        config=config)
    
    assert all([x in trees.columns for x in ["tile","geometry","ens_score","ensembleTaxonID"]])
    output_dict = pd.read_csv(os.path.join(tmpdir,"2019_D01_HARV_DP3_726000_4699000_image_crop_2018_0.csv"))
    assert output_dict.shape[0] == crowns.shape[0]