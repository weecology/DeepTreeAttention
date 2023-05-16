import pytest
import geopandas as gpd
import os

from src import predict
from src.models import dead, multi_stage
from src import utils
from pytorch_lightning import Trainer
import cProfile
import pstats
from pytorch_lightning import Trainer

#Training module
@pytest.fixture()
def species_model_path(m, tmpdir):
    trainer = Trainer(fast_dev_run=True)
    for key in m.level_names:
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
    trainer.save_checkpoint("{}/model.pl".format(tmpdir))
    
    return "{}/model.pl".format(tmpdir)
    
def test_predict_tile(species_model_path, config, ROOT, tmpdir):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop_2019.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    os.makedirs("{}/prediction".format(tmpdir), exist_ok=True)
    os.makedirs("{}/crowns/".format(tmpdir), exist_ok=True)        
    os.makedirs("{}/crops/".format(tmpdir), exist_ok=True)    
    os.makedirs("{}/crops/tar/".format(tmpdir), exist_ok=True)
    os.makedirs("{}/crops/shp/".format(tmpdir), exist_ok=True)
    
    rgb_pool, hsi_pool, h5_pool, CHM_pool = utils.create_glob_lists(config)
    dead_model = dead.AliveDead(config)
    trainer = Trainer(fast_dev_run=True)    
    trainer.fit(dead_model)
    dead_model_path = "{}/dead_model.pl".format(tmpdir)
    trainer.save_checkpoint(dead_model_path)
    
    crown_path = predict.find_crowns(
        rgb_path,
        config,
        dead_model_path=dead_model_path,
        savedir="{}/crowns/".format(tmpdir)
    )
    crowns = gpd.read_file(crown_path)    
    assert len(crowns.individual.unique()) == crowns.shape[0]
            
    crown_annotations_path = predict.generate_prediction_crops(
        crown_path=crown_path,
        config=config,
        rgb_pool=rgb_pool,
        h5_pool=h5_pool,
        img_pool=hsi_pool,
        crop_dir="{}/crops/".format(tmpdir))
    
    crown_annotations = gpd.read_file(crown_annotations_path)
    
    # Assert that the geometry is correctly mantained
    assert crown_annotations.iloc[0].geometry.bounds == crowns[crowns.individual==crown_annotations.iloc[0].individual].iloc[0].geometry.bounds
    assert all(~crown_annotations.score.isnull())
        
    #There should be two of each individual, with the same geoemetry
    assert crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].shape[0] == 2
    assert len(crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].bounds.minx.unique()) == 1
    
    m = multi_stage.MultiStage.load_from_checkpoint(species_model_path, config=config)
    trees = predict.predict_tile(
        crown_annotations=crown_annotations_path,
        m=m,
        site="pytest",
        trainer=trainer,
        filter_dead=True,
        savedir=tmpdir,
        config=config)
    
    assert all([x in trees.columns for x in ["geometry","crown_score","scientificName"]])
    assert all(~trees.crown_score.isnull())