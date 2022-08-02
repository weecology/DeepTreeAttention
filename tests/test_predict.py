import os
import pytest
from src import predict
from src.models import multi_stage
from pytorch_lightning import Trainer
import geopandas as gpd
import pandas as pd
import cProfile
import pstats

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
    config["prediction_crop_dir"] = tmpdir    
    
    crowns = predict.find_crowns(rgb_path, config)
    crowns.to_file("{}/crowns.shp".format(tmpdir))
    
    crown_annotations_path = predict.generate_prediction_crops(crowns, config)
    
    profiler = cProfile.Profile()
    profiler.enable()    
    
    trees = predict.predict_tile(
        crown_annotations=crown_annotations_path,
        filter_dead=False,
        species_model_path=species_model_path,
        savedir=tmpdir,
        config=config)
    
    profiler.disable()
    profiler.dump_stats("{}/tests/predict_profile.prof".format(ROOT))
    stats = pstats.Stats(profiler).sort_stats('cumtime')        
    stats.print_stats()
    
    
    assert all([x in trees.columns for x in ["tile","geometry","ens_score","ensembleTaxonID"]])
    output_dict = pd.read_csv(os.path.join(tmpdir,"2019_D01_HARV_DP3_726000_4699000_image_crop_2018_0.csv"))
