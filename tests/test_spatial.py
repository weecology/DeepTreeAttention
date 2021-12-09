#test spatial neighbors
from src import spatial
import geopandas as gpd
import numpy as np
import os

ROOT = os.path.dirname(os.path.dirname(spatial.__file__))
def test_spatial_neighbors():
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    neighbors = spatial.spatial_neighbors(gdf, buffer=15)
    assert len(neighbors) == gdf.shape[0]
    assert len(neighbors[0]) == 1
    
def test_spatial_smooth():
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    neighbors = spatial.spatial_neighbors(gdf, buffer=15)
    
    #Create a set of features
    features = np.zeros((gdf.shape[0],len(gdf.taxonID.unique())))
    features[0,1] = 0.95
    features[1,1] = 0.75
    
    spatial_features = spatial.spatial_smooth(neighbors, features)
    assert spatial_features.shape == features.shape
    
    