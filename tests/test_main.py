import comet_ml
import geopandas as gpd
from src import main
from src.models import Hang2020
from src import data
import numpy as np
import os
import rasterio
import pytest
import pandas as pd
from pytorch_lightning import Trainer
import tempfile

ROOT = os.path.dirname(os.path.dirname(data.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@pytest.fixture(scope="session")
def config():
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["bands"] = 3
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    
    return config
#Data module
@pytest.fixture(scope="session")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm


@pytest.fixture()
def comet_experiment():
    if not "GITHUB_ACTIONS" in os.environ:
        from pytorch_lightning.loggers import CometLogger        
        COMET_KEY = os.getenv("COMET_KEY")
        comet_logger = CometLogger(api_key=COMET_KEY,
                                   project_name="DeepTreeAttention", workspace="bw4sz",auto_output_logging = "simple")
        return comet_logger
    else:
        return None


#Training module
@pytest.fixture(scope="session")
def m(config, dm):
    m = main.TreeModel(model=Hang2020.vanilla_CNN, bands=3, classes=2, config=config, label_dict=dm.species_label_dict)
    
    return m

def test_fit(config, m, dm):
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m,datamodule=dm)
    
def test_predict_file(config, m, comet_experiment):
    df = m.predict_file("{}/tests/data/processed/test.csv".format(ROOT), experiment = comet_experiment.experiment)
    input_data = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))    
    
    assert df.shape[0] == len(input_data.image_path.apply(lambda x: os.path.basename(x).split("_")[0]).unique())

def test_evaluate_crowns(config, m, comet_experiment):
    df = m.evaluate_crowns("{}/tests/data/processed/test.csv".format(ROOT))
    
    assert len(df) == 2

def test_predict_xy(config, m, dm):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    df = pd.read_csv(csv_file)
    label, score = m.predict_xy(coordinates=(df.itcEasting[0],df.itcNorthing[0]))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 

def test_predict_crown(config, m, dm):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    label, score = m.predict_crown(geom = gdf.geometry[0], sensor_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT))
    
    assert label in dm.species_label_dict.keys()
    assert score > 0 
    
    
def test_predict_raster(dm, m, config, tmpdir):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    src = rasterio.open(rgb_path)
    
    #Make smaller crop
    img = src.read(window = rasterio.windows.Window(col_off=0, row_off=0, width = 100, height=100), boundless=False)
    with rasterio.open("{}/test_prediction.tif".format(tmpdir), "w", driver="GTiff",height=100, width=100, count = 3, dtype=img.dtype) as dst:
        dst.write(img)
        
    prediction_raster = m.predict_raster(raster_path="{}/test_prediction.tif".format(tmpdir))
    
    assert prediction_raster.shape == (100,100)
    
    #Pixels should be either 0, 1 for the 2 class labels
    assert all(np.unique(prediction_raster) == [0,1])
