#test main module
from src import main
from src.models import Hang2020
from src import data
import os
import rasterio
import pytest
import pandas as pd
from pytorch_lightning import Trainer

ROOT = os.path.dirname(os.path.dirname(data.__file__))

@pytest.fixture()
def config(tmpdir):
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tmpdir
    config["bands"] = 3
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    
    return config

#Data module
@pytest.fixture()
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=False, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm

#Training module
@pytest.fixture()
def m(config, dm):
    m = main.TreeModel(model=Hang2020.vanilla_CNN, bands=3, classes=2, config=config, label_dict=dm.species_label_dict)
    
    return m

def test_fit(config, m, dm):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)
    
def test_predict_file(config, m):
    df = m.predict_file("{}/tests/data/processed/test.csv".format(ROOT))
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert set(df.columns) == set(["crown","label"])
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]).unique())

def test_evaluate_crowns(config, m):
    df = m.evaluate_crowns("{}/tests/data/processed/test.csv".format(ROOT))
    
    assert len(df) == 2

def test_predict_xy(config, m, dm):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    df = pd.read_csv(csv_file)
    label, score = m.predict_xy(coordinates=(df.itcEasting[0],df.itcNorthing[0]))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 

def test_predict_raster(dm, m, config):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    prediction_raster = m.predict_raster(raster_path=rgb_path)
    
    r = rasterio.open(rgb_path).read()
    assert r.shape == prediction_raster.shape
    
    #Pixels should be either 0, 1 for the 2 class labels
    for x in prediction_raster:
        assert x in [0,1]