#Test data module
from src import data
import pytest
import pandas as pd
import tempfile
import numpy as np

import os
ROOT = os.path.dirname(os.path.dirname(data.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

@pytest.fixture(scope="module")
def config():
    #Turn off CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_test_samples"] = 1
    config["min_train_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["convert_h5"] = False
    config["megaplot_dir"] = None
    return config


#Data module
@pytest.fixture(scope="module")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)            
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=True, data_dir="{}/tests/data".format(ROOT), debug=True) 
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
    assert all([x in ["image_path","label","site","taxonID","siteID","plotID","individualID","point_id","box_id"] for x in train.columns])
    
def test_TreeDataset(dm, config,tmpdir):
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/train.csv".format(ROOT), config=config, image_size=config["image_size"])
    individuals, inputs, label = data_loader[0]
    image = inputs["HSI"]
    assert image.shape == (3, config["image_size"], config["image_size"])
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/tests/data/processed/test.csv".format(ROOT), train=False, config=config)    
    annotations = pd.read_csv("{}/tests/data/processed/test.csv".format(ROOT))
    
    assert len(data_loader) == annotations.shape[0]
