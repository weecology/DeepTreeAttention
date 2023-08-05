#test patches
import os
from src import patches
from src import __file__ as ROOT
import geopandas as gpd
import rasterio

ROOT = os.path.dirname(os.path.dirname(ROOT))   
    
def test_crop(tmpdir, rgb_path, sample_crowns):
    gdf = gpd.read_file(sample_crowns)
    patch = patches.crop(
        bounds=gdf.geometry[0].bounds,
        sensor_path=rgb_path,
        savedir=tmpdir,
        basename="test")
    img = rasterio.open(patch).read()
    assert img.shape[0] == 3    