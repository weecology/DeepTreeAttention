#test metrics
from src import metrics
from src.models import Hang2020
from src import data
import os
import pandas as pd
import pytest
import tempfile

ROOT = os.path.dirname(os.path.dirname(data.__file__))
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def test_site_confusion():
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]
    site_lists = {0:[0], 1:[0]}
    assert metrics.site_confusion(y_true, y_pred, site_lists) == 1
    
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]
    site_lists = {0:[1], 1:[0]}
    assert metrics.site_confusion(y_true, y_pred, site_lists) == 0    
    
def test_genus_confusion():
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,2]   
    
    scientific_dict = {0:"ACRU",1:"QUMU",2:"QUMLA"}
    
    #all error is within genus
    assert metrics.genus_confusion(y_true, y_pred, scientific_dict) == 1
    
    y_true = [0,0,1,1,1]
    y_pred = [0,0,1,1,0]   
    
    scientific_dict = {0:"ACRU",1:"QUMU",2:"QUMLA"}
    
    #all error is outside genus
    assert metrics.genus_confusion(y_true, y_pred, scientific_dict) == 0  
    

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
    config["plot_n_individuals"] = 1
    
    return config

#Data module
@pytest.fixture(scope="session")
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=regen, data_dir="{}/tests/data".format(ROOT), metadata=False) 
    dm.setup()    
    
    return dm

#Training module
@pytest.fixture(scope="session")
def m(config, dm):
    m = Hang2020.vanilla_CNN(bands=3, classes=2)
    
    return m


def test_novel_prediction(config, m):
    df = metrics.novel_prediction(model=m, csv_file="{}/tests/data/processed/train.csv".format(ROOT), config=config)
    original = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    
    assert original.shape[0] == df.shape[0]
    assert all([x in df.columns for x in ["individualID","top_score","softmax_score","taxonID"]])
                                         