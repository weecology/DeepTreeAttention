#test patches
import os
from src import patches
from src import __file__ as ROOT
import numpy as np
import geopandas as gpd
import rasterio

ROOT = os.path.dirname(os.path.dirname(ROOT))

def test_crop_hsi(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch = patches.crop(bounds=gdf.geometry[0].bounds,sensor_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT), savedir=tmpdir, basename="test")
    img = rasterio.open(patch).read()
    assert img.shape[0] == 369