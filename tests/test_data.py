#Test data module
from src import data
from src import generate
import glob
import geopandas as gpd
import pytest

import os
from distributed import Client
ROOT = os.path.dirname(os.path.dirname(data.__file__))


@pytest.fixture()
def config(tmpdir):
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tmpdir
    
    return config

def test_TreeData_setup(config):
    #One site's worth of data
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
    data_module.setup(csv_file=csv_file, regenerate=True, client=None)
    
def test_TreeDataset(tmpdir):
    data_path = "{}/tests/data/crown.shp".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))
    gdf = gpd.read_file(data_path)
    annotations = generate.generate_crops(gdf=gdf, rgb_pool=rgb_pool, crop_save_dir=tmpdir, label_dict={"ACRU":0,"BELE":1})   
    annotations.to_csv("{}/train.csv".format(tmpdir))
    
    #Train loader
    data_loader = data.TreeDataset(csv_file="{}/train.csv".format(tmpdir))
    image, label = data_loader[0]
    assert len(data_loader) == annotations.shape[0]
    
    #Test loader
    data_loader = data.TreeDataset(csv_file="{}/test.csv".format(tmpdir))
    image = data_loader[0]
    assert len(data_loader) == annotations.shape[0]    
    
    
def test_TreeData_evaluate(tmpdir):
    #dask client to test
    data_module = data.TreeData(config=config, data_dir="{}/tests/data".format(ROOT))
