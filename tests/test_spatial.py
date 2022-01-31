#test spatial neighbors
from src import spatial
from src.generate import run
from src import main
from src import utils
from src.models import Hang2020
import tempfile
import glob
import geopandas as gpd
import numpy as np
import os
import pytest

#Make sure deepforect crown boxes exist
@pytest.fixture()
def deepforest_boxes(ROOT):
    df = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))        
    for plot in df.plotID.unique():
        run(plot=plot, df=df, savedir=None, raw_box_savedir="{}/tests/data/interim/".format(ROOT), rgb_pool=rgb_pool)

def test_spatial_neighbors(deepforest_boxes, config, ROOT):
    gdf = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    m =  Hang2020.vanilla_CNN(bands=3, classes=5)
    #Taxon to ID dict and the reverse    
    species_label_dict = {}
    for index, taxonID in enumerate(gdf.taxonID.unique()):
        species_label_dict[taxonID] = index    
    m = main.TreeModel(model=m, classes=5, config=config, label_dict=species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))            
    neighbors = spatial.spatial_neighbors(gdf, buffer=4, HSI_pool=rgb_pool, data_dir="{}/tests/data/".format(ROOT), model=m, image_size=11)
    assert len(neighbors) == gdf.shape[0]
    
def test_spatial_smooth(deepforest_boxes, config, ROOT):
    gdf = gpd.read_file("{}/tests/data/sample.shp".format(ROOT))
    m =  Hang2020.vanilla_CNN(bands=3, classes=5)
    #Taxon to ID dict and the reverse    
    species_label_dict = {}
    for index, taxonID in enumerate(gdf.taxonID.unique()):
        species_label_dict[taxonID] = index    
    m = main.TreeModel(model=m, classes=5, config=config, label_dict=species_label_dict)
    m.ROOT = "{}/tests/".format(ROOT)
    rgb_pool = glob.glob("{}/tests/data/*.tif".format(ROOT))            
    neighbors = spatial.spatial_neighbors(gdf, buffer=4, HSI_pool=rgb_pool, data_dir="{}/tests/data/".format(ROOT), model=m, image_size=11)
    
    #Create a set of features
    features = np.zeros((gdf.shape[0],len(gdf.taxonID.unique())))
    features[0,1] = 0.95
    features[1,1] = 0.75    
    labels, scores = spatial.spatial_smooth(neighbors, features)
    assert len(labels) == gdf.shape[0]
    assert len(scores) == gdf.shape[0]
    