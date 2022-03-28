#Download deepforest before tests start
import comet_ml
from deepforest.main import deepforest
import geopandas as gpd
import os
import glob
import rasterio as rio
from src import data
from src.models import Hang2020
from src.models import dead
from src import main
from src import utils
import tempfile
import torch
from pytorch_lightning import Trainer
import pandas as pd
import pytest

#Set Env VARS
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def pytest_sessionstart():
    # prepare something ahead of all tests
    m = deepforest()
    m.use_release()    

@pytest.fixture(scope="session")
def ROOT():
    ROOT = os.path.dirname(os.path.dirname(data.__file__))
    
    return ROOT

@pytest.fixture(scope="session")
def rgb_pool(ROOT):
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
    
    return rgb_pool

@pytest.fixture(scope="session")
def rgb_path(ROOT):
    rgb_path = "{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT)
    
    return rgb_path

@pytest.fixture(scope="session")
def sample_crowns(ROOT):
    data_path = "{}/tests/data/sample.shp".format(ROOT)
    
    return data_path

@pytest.fixture(scope="session")
def plot_data(ROOT, sample_crowns):
    plot_data = gpd.read_file(sample_crowns)        
    
    return plot_data

@pytest.fixture(scope="session")
def config(ROOT):
    print("Creating global config")
    #Turn of CHM filtering for the moment
    config = utils.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_train_samples"] = 1
    config["min_test_samples"] = 1
    config["crop_dir"] = "{}/tests/data/crops/".format(ROOT)
    config["bands"] = 3
    config["classes"] = 3
    config["top_k"] = 1
    config["convert_h5"] = False
    config["plot_n_individuals"] = 1
    config["min_CHM_diff"] = None    
    config["dead_model"] = None
    config["dead_threshold"] = 0.95
    config["megaplot_dir"] = None
    config["use_data_commit"] = None
    config["dead"]["epochs"] = 1
    
    return config

#Data module
@pytest.fixture(scope="session")
def dm(config, ROOT):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)               
    dm = data.TreeData(config=config, csv_file=csv_file, data_dir="{}/tests/data".format(ROOT), debug=True, metadata=True) 
    dm.setup()    
    
    return dm

@pytest.fixture(scope="session")
def comet_logger():
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
def m(config, dm, ROOT):
    model = Hang2020.spectral_network(bands=3, classes=3)
    m = main.TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict, loss_weight=[0.1,0.8,1])
    m.ROOT = "{}/tests/".format(ROOT)
    
    return m

#Training module
@pytest.fixture(scope="session")
def species_model_path(config, dm):
    model = Hang2020.spectral_network(bands=3, classes=3)
    m = main.TreeModel(model=model, classes=3, config=config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    filepath = "{}/model.pl".format(tempfile.gettempdir())
    trainer = Trainer(fast_dev_run=True)
    trainer.fit(m, dm)
    trainer.save_checkpoint(filepath)
    
    return filepath