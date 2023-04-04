import pytest
import geopandas as gpd

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
    config["prediction_crop_dir"] = tmpdir    
    
    rgb_pool, hsi_pool, h5_pool, CHM_pool = utils.create_glob_lists(config)
    dead_model = dead.AliveDead(config)
    trainer = Trainer(fast_dev_run=True)    
    trainer.fit(dead_model)
    dead_model_path = "{}/dead_model.pl".format(tmpdir)
    trainer.save_checkpoint(dead_model_path)
    
    crowns = predict.find_crowns(rgb_path, config, dead_model_path=dead_model_path)
    assert len(crowns.individual.unique()) == crowns.shape[0]
        
    crowns.to_file("{}/prediction_crowns.shp".format(tmpdir))
    
    crown_annotations_path = predict.generate_prediction_crops(crown_path="{}/prediction_crowns.shp".format(tmpdir), config=config,
                                                               rgb_pool=rgb_pool, h5_pool=h5_pool, img_pool=hsi_pool, crop_dir=tmpdir)
    crown_annotations = gpd.read_file(crown_annotations_path)
    
    # Assert that the geometry is correctly mantained
    assert crown_annotations.iloc[0].geometry.bounds == crowns[crowns.individual==crown_annotations.iloc[0].individual].iloc[0].geometry.bounds
    
    #There should be two of each individual, with the same geoemetry
    assert crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].shape[0] == 2
    assert len(crown_annotations[crown_annotations.individual == crown_annotations.iloc[0].individual].bounds.minx.unique()) == 1
    
    trees = predict.predict_tile(
        crown_annotations=crown_annotations_path,
        model_path=species_model_path,
        filter_dead=True,
        savedir=tmpdir,
        config=config)
    
    assert all([x in trees.columns for x in ["geometry","score","pred_taxa_top1"]])
    assert trees.iloc[0].geometry.bounds == trees[trees.individual==trees.iloc[0].individual].iloc[1].geometry.bounds
