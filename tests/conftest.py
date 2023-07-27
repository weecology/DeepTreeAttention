#Download deepforest before tests start
import comet_ml
from deepforest.main import deepforest
import geopandas as gpd
import os
import glob
from src import data
from src.models import dead, multi_stage, Hang2020
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
    rgb_pool = glob.glob("{}/tests/data/HARV/2019/FullSite/D01/2019_HARV_6/L3/Camera/Mosaic/*.tif".format(ROOT))
    
    return rgb_pool

@pytest.fixture(scope="session")
def rgb_path(ROOT):
    rgb_path = "{}/tests/data/HARV/2019/FullSite/D01/2019_HARV_6/L3/Camera/Mosaic/2019_D01_HARV_DP3_726000_4699000_image.tif".format(ROOT)
    
    return rgb_path

@pytest.fixture(scope="session")
def sample_crowns(ROOT):
    data_path = "{}/tests/data/sample_crowns.shp".format(ROOT)
    
    return data_path

@pytest.fixture(scope="session")
def plot_data(ROOT, sample_crowns):
    plot_data = gpd.read_file(sample_crowns)        
    
    return plot_data

@pytest.fixture(scope="session")
def config(ROOT, tmpdir_factory):
    print("Creating global config")
    #Turn of CHM filtering for the moment
    config = utils.read_config(config_path="{}/config.yml".format(ROOT))
    config["use_data_commit"] = None
    config["replace_bounding_boxes"] = True
    config["replace_crops"] = True
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["HSI_tif_dir"] = "{}/tests/data/hsi/".format(ROOT)
    config["min_train_samples"] = 1
    config["min_test_samples"] = 1
    config["bands"] = 3
    config["classes"] = 3
    config["top_k"] = 1
    config["head_class_minimum_ratio"] = 0.25
    config["convert_h5"] = False
    config["min_CHM_diff"] = None    
    config["dead_model"] = None
    config["dead_threshold"] = 0.95
    config["megaplot_dir"] = None
    config["dead"]["epochs"] = 1
    config["pretrain_state_dict"] = None
    config["preload_images"] = False
    config["batch_size"] = 1
    config["gpus"] = 0
    config["existing_test_csv"] = None
    config["workers"] = 0
    config["dead"]["num_workers"] = 0
    config["dead"]["batch_size"] = 2
    config["fast_dev_run"] = True
    config["snapshot_dir"] = None
    config["taxonomic_csv"] = "{}/data/raw/families.csv".format(ROOT)
    config["crop_dir"] = tmpdir_factory.mktemp('data')
    config["data_dir"] = tmpdir_factory.mktemp('output')
    config["rgb_sensor_pool"] = "{}/tests/data/HARV/2019/FullSite/D01/2019_HARV_6/L3/Camera/Mosaic/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/HARV/2019/FullSite/D01/2019_HARV_6/L3/Camera/Mosaic/*.tif".format(ROOT)

    return config

#Data module
@pytest.fixture(scope="session")
def dm(config, ROOT):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(
        config=config,
        csv_file=csv_file,
        data_dir=config["crop_dir"],
        site="HARV",
        create_train_test=True) 
    
    return data_module

@pytest.fixture(scope="session")
def experiment():
    if not "GITHUB_ACTIONS" in os.environ:
        from pytorch_lightning.loggers import CometLogger        
        COMET_KEY = os.getenv("COMET_KEY")
        comet_logger = CometLogger(api_key=COMET_KEY,
                                   project_name="DeepTreeAttention2", workspace="bw4sz",auto_output_logging = "simple")
        return comet_logger.experiment
    else:
        return None

#Training module
@pytest.fixture(scope="session")
def m(config, dm, ROOT):
    m = multi_stage.MultiStage(train_df=dm.train, test_df=dm.test, config=config)
    m.ROOT = "{}/tests/".format(ROOT)
    m.setup("fit")
    
    return m