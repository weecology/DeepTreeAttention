#test spatial neighbors
from src import spatial
from src.generate import run
from src import main
from src import data
from src.models import Hang2020
import tempfile
import glob
import geopandas as gpd
import numpy as np
import os
import pytest
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(spatial.__file__))

@pytest.fixture()
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
    config["megaplot_dir"] = None
    config["plot_n_individuals"] = None
    
    return config

#Data module
@pytest.fixture()
def dm(config):
    config = data.read_config(config_path="{}/config.yml".format(ROOT))    
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    dm = data.TreeData(config=config, csv_file=csv_file, regenerate=True, data_dir="{}/tests/data".format(ROOT)) 
    dm.setup()    
    
    return dm

#Make sure deepforect crown boxes exist
@pytest.fixture()
def deepforest_boxes():
    df = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))        
    for plot in df.plotID.unique():
        run(plot=plot, df=df, savedir=None, raw_box_savedir="{}/tests/data/interim/".format(ROOT), rgb_pool=rgb_pool)

def test_spatial_neighbors(deepforest_boxes, dm):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    m =  Hang2020.vanilla_CNN(bands=3, classes=3)
    m = main.TreeModel(model=m, classes=3, config=dm.config, label_dict=dm.species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))            
    neighbors = spatial.spatial_neighbors(gdf, buffer=15, rgb_pool=rgb_pool, data_dir="{}/tests/data/interim/".format(ROOT), model=m)
    assert len(neighbors) == gdf.shape[0]
    assert len(neighbors[1]) == 1
    
def test_spatial_smooth(deepforest_boxes):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    neighbors = spatial.spatial_neighbors(gdf, buffer=15)
    
    #Create a set of features
    features = np.zeros((gdf.shape[0],len(gdf.taxonID.unique())))
    features[0,1] = 0.95
    features[1,1] = 0.75    
    labels, score = spatial.spatial_smooth(neighbors, features)
    assert score[0] == 0.95 + (0.2 * 0.75)
    assert score[1] == 0.75 + (0.2 * 0.95)
    assert labels[0] == 1
    assert labels[1] == 1    
    