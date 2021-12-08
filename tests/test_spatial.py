#test spatial neighbors
from src import spatial
import geopandas as gpd
import os

ROOT = os.path.dirname(os.path.dirname(spatial.__file__))
def test_spatial_neighbors():
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    neighbors = spatial.spatial_neighbors(gdf, buffer=3)
    assert len(neighbors) == gdf.shape[0]