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
    config["convert_h5"] = False
    
    return config

def test_TreeData_setup(config):
    #One site's worth of data
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT), csv_file=csv_file, regenerate=True, client=None)
    data_module.setup()
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    
def test_TreeDataset(config,tmpdir):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT), csv_file=csv_file, regenerate=False, client=None)
    data_module.setup()

    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/train.csv".format(ROOT))
    image, label = data_loader[0]
    assert image.shape == (3, config["window_size"], config["window_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/test.csv".format(ROOT), train=False)
    image = data_loader[0]
    assert image.shape == (3, config["window_size"], config["window_size"])
    
    annotations = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    assert len(data_loader) == annotations.shape[0]    
    
def test_resample(config, tmpdir):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT), csv_file=csv_file, regenerate=False, client=None)
    data_module.setup()
    #Set to a smaller number to ensure easy calculation
    data_module.config["resample_max"] = 50
    data_module.config["resample_min"] = 10
    
    annotations = data_module.resample(csv_file = "{}/tests/data/processed/train.csv".format(ROOT), oversample=False)
    
    #There are two classes
    assert annotations.shape[0] == data_module.config["resample_max"] * 2
    
    undersampling = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    undersampling = undersampling.groupby("label").sample(n=5)
    undersampling.to_csv("{}/undersampling.csv".format(tmpdir))
    annotations = data_module.resample(csv_file="{}/undersampling.csv".format(tmpdir), oversample=True)
    
    #There are two classes
    assert annotations.shape[0] == 2 * data_module.config["resample_min"]
