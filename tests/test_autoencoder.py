#test autoencoder
from src.models import autoencoder
from src import data
import torch
import os
import pytest
import pandas as pd
import tempfile
os.environ['KMP_DUPLICATE_LIB_OK']='True'

ROOT = os.path.dirname(os.path.dirname(data.__file__))

@pytest.fixture(scope="session")
def config():
    #Turn off CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["convert_h5"] = False
    config["megaplot_dir"] = None
    config["gpus"] = 0
    config["fast_dev_run"] = True
    config["bands"] = 3
    
    return config

def test_conv_module():
    m = autoencoder.conv_module(in_channels=369, filters=32)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,32,11,11)

def test_autoencoder(config):
    m = autoencoder.autoencoder(bands=369, classes=10, config=config, data_dir="{}/tests/".format(ROOT))
    image = torch.randn(20, 369, 11, 11)
    output = m(image)    
    assert output.shape == image.shape
    
def test_find_outliers(config):
    prediction = autoencoder.find_outliers(data_dir="{}/tests/".format(ROOT), classes=2, config=config)
    
    prediction = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    assert not prediction.empty
    assert prediction.columns == ["individual","loss"]

    