#test patches
import os
from src import patches
from src import __file__ as ROOT
import geopandas as gpd
import rasterio

ROOT = os.path.dirname(os.path.dirname(ROOT))

def test_patches_rgb(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch_list = patches.bounds_to_pixel(bounds=gdf.geometry[0].bounds,img_path="{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT), savedir=tmpdir, basename="test")
    assert len(patch_list) > 0 
    img = rasterio.open(patch_list[0]).read()
    assert img.shape == (3,11,11)
    
def test_patches_hsi(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch_list = patches.bounds_to_pixel(bounds=gdf.geometry[0].bounds,img_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT), savedir=tmpdir, basename="test")
    assert len(patch_list) > 0 
    img = rasterio.open(patch_list[0]).read()
    assert img.shape == (369,11,11)    