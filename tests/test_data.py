#Test data module
from src import data
from src import generate
import glob
import geopandas as gpd
import pytest
import pandas as pd

import os
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
    
    return config

def test_TreeData_setup(config):
    #One site's worth of data
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
    data_module.setup(csv_file=csv_file, regenerate=True, client=None)
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any(test.image_path.unique() == train.image_path.unique())
    
def test_TreeDataset(config,tmpdir):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
    data_module.setup(csv_file=csv_file, regenerate=True, client=None)

    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/train.csv".format(ROOT))
    image, label = data_loader[0]
    assert image.shape == (config["window_size"], config["window_size"],3)
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/test.csv".format(ROOT), train=False)
    image = data_loader[0]
    assert image.shape == (config["window_size"], config["window_size"],3)
    
    annotations = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    assert len(data_loader) == annotations.shape[0]    
