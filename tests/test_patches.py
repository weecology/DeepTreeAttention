#test patches
import os
from src import patches
from src import __file__ as ROOT
import geopandas as gpd

ROOT = os.path.dirname(os.path.dirname(ROOT))

def test_patches():
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch_list = patches.crown_to_pixel(crown=gdf.geometry[0],img_path="{}/tests/data/2019_D01_HARV_DP3_726000_4699000_image_crop.tif".format(ROOT))
    assert len(patch_list) > 0 
    assert patch_list[0].shape == (3,11,11)