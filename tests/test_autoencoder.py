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
    config["fast_dev_run"] = False
    config["autoencoder_epochs"] = 1
    config["bands"] = 3
    config["outlier_threshold"] = 0.5
    config["batch_size"] = 2
    config["top_k"] = 1
    
    return config

def test_conv_module():
    m = autoencoder.conv_module(in_channels=369, filters=32)
    image = torch.randn(20, 369, 11, 11)
    output = m(image)
    assert output.shape == (20,32,11,11)

def test_autoencoder(config):
    m = autoencoder.autoencoder(csv_file="{}/tests/data/train.csv".format(ROOT), classes=10, bands=369, config=config, data_dir="{}/tests/".format(ROOT))
    images = torch.randn(20, 369, 11, 11)
    labels = torch.randint(0, 10, (20,))
    
    autoencoder_output, classification_output = m(images)    
    assert autoencoder_output.shape == images.shape
    assert classification_output.shape == (20, 10)
    
def test_find_outliers(config):
    df = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    prediction = autoencoder.find_outliers(annotations=df, data_dir="{}/tests/data/".format(ROOT), config=config)
    assert all(prediction.columns == ["individual","loss"])

    