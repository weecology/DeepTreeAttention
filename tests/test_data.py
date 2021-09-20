#Test data module
from src import data
import pytest
import pandas as pd
import tempfile

import os
ROOT = os.path.dirname(os.path.dirname(data.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

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
    
    return config


#Data module
@pytest.fixture(scope="session")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=True, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm

def test_TreeData_setup(dm, config):
    #One site's worth of data
    dm.setup()
    test = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    train = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert not test.empty
    assert not train.empty
    assert not any([x in train.image_path.unique() for x in test.image_path.unique()])
    
def test_TreeDataset(dm, config,tmpdir):
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/train.csv".format(ROOT), config=config)
    inputs, label = data_loader[0]
    image = inputs["HSI"]
    assert image.shape == (3, config["image_size"], config["image_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/test.csv".format(ROOT), train=False)    
    annotations = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    assert len(data_loader) == annotations.shape[0]    
    
#def test_resample(config, dm, tmpdir):
    ##Set to a smaller number to ensure easy calculation
    #dm.config["resample_max"] = 50
    #dm.config["resample_min"] = 10
    
    #annotations = dm.resample(csv_file = "{}/tests/data/processed/train.csv".format(ROOT), oversample=False)
    
    ##There are two classes
    #assert annotations.shape[0] == dm.config["resample_max"] * 2
    
    #undersampling = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    #undersampling = undersampling.groupby("label").sample(n=5)
    #undersampling.to_csv("{}/undersampling.csv".format(tmpdir))
    #annotations = dm.resample(csv_file="{}/undersampling.csv".format(tmpdir), oversample=True)
    
    ##There are two classes
    #assert annotations.shape[0] == 2 * dm.config["resample_min"]
