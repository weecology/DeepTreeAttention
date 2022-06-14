import os
import pytest
from src import predict
from src.models import dead
from src.models import multi_stage
from pytorch_lightning import Trainer
import cProfile, pstats

#Training module
@pytest.fixture()
def species_model_path(config, dm, ROOT, tmpdir):
    config["batch_size"] = 16    
    m  = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, crowns=dm.crowns, config=config)    
    m.ROOT = "{}/tests/".format(ROOT)
    trainer = Trainer(fast_dev_run=False, max_epochs=1)
    trainer.fit(m)
    trainer.save_checkpoint("{}/model.pl".format(tmpdir))
    
    return "{}/model.pl".format(tmpdir)
    
def test_predict_tile(species_model_path, config, ROOT, tmpdir):
    HSI_paths = {}
    HSI_paths["2019"] = "{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/hsi/*.tif".format(ROOT)
    config["CHM_pool"] = None
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    trees = predict.predict_tile(
        HSI_paths=HSI_paths,
        dead_model_path=None,
        species_model_path=species_model_path,
        savedir=tmpdir,
        config=config)
    
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('ncalls')
    stats.print_stats()

    assert all([x in trees.columns for x in ["tile","geometry","ens_score","ensembleTaxonID"]])
    