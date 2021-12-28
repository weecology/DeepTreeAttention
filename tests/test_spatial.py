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
    config["classes"] = 5
    config["top_k"] = 1
    config["convert_h5"] = False
    config["megaplot_dir"] = None
    config["plot_n_individuals"] = None
    
    return config

#Make sure deepforect crown boxes exist
@pytest.fixture()
def deepforest_boxes():
    df = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))        
    for plot in df.plotID.unique():
        run(plot=plot, df=df, savedir=None, raw_box_savedir="{}/tests/data/interim/".format(ROOT), rgb_pool=rgb_pool)

def test_spatial_neighbors(deepforest_boxes, config):
    gdf = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    m =  Hang2020.vanilla_CNN(bands=3, classes=5)
    #Taxon to ID dict and the reverse    
    species_label_dict = {}
    for index, taxonID in enumerate(gdf.taxonID.unique()):
        species_label_dict[taxonID] = index    
    m = main.TreeModel(model=m, classes=5, config=config, label_dict=species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))            
    neighbors = spatial.spatial_neighbors(gdf, buffer=4, rgb_pool=rgb_pool, data_dir="{}/tests/data/".format(ROOT), model=m, image_size=11)
    assert len(neighbors) == gdf.shape[0]
    
def test_spatial_smooth(deepforest_boxes, config):
    gdf = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    m =  Hang2020.vanilla_CNN(bands=3, classes=5)
    #Taxon to ID dict and the reverse    
    species_label_dict = {}
    for index, taxonID in enumerate(gdf.taxonID.unique()):
        species_label_dict[taxonID] = index    
    m = main.TreeModel(model=m, classes=5, config=config, label_dict=species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))            
    neighbors = spatial.spatial_neighbors(gdf, buffer=4, rgb_pool=rgb_pool, data_dir="{}/tests/data/".format(ROOT), model=m, image_size=11)
    
    #Create a set of features
    features = np.zeros((gdf.shape[0],len(gdf.taxonID.unique())))
    features[0,1] = 0.95
    features[1,1] = 0.75    
    labels, scores = spatial.spatial_smooth(neighbors, features)
    assert len(labels) == gdf.shape[0]
    assert len(scores) == gdf.shape[0]
    