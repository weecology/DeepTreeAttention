#test patches
import os
from src import patches
from src import __file__ as ROOT
import numpy as np
import geopandas as gpd
import rasterio

ROOT = os.path.dirname(os.path.dirname(ROOT))


# Masked of a polygon is different than non-masked
def test_crop_hsi_mask(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch = patches.crop(
        bounds=gdf.geometry[2].bounds,
        sensor_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT),
        savedir=tmpdir,
        basename="test")
    img_no_mask = rasterio.open(patch).read()

    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch = patches.crop(
        bounds=gdf.geometry[2].bounds,
        sensor_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT),
        savedir=tmpdir,
        basename="test",
        mask=gdf.geometry[2]
    )
    img_masked = rasterio.open(patch).read()
    
    assert not np.array_equal(img_no_mask, img_masked)
    assert not np.sum(img_masked) == 0
    
# Masked of a swuare crop is the same as non-masked
def test_crop_hsi(tmpdir):
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch = patches.crop(
        bounds=gdf.geometry[0].bounds,
        sensor_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT),
        savedir=tmpdir,
        basename="test")
    img_no_mask = rasterio.open(patch).read()
    
    gdf = gpd.read_file("{}/tests/data/crown.shp".format(ROOT))
    patch = patches.crop(
        bounds=gdf.geometry[0].bounds,
        sensor_path="{}/tests/data/hsi/2019_HARV_6_726000_4699000_image_crop_hyperspectral.tif".format(ROOT),
        savedir=tmpdir,
        basename="test",
        mask=gdf.geometry[0]
    )
    img_masked = rasterio.open(patch).read()
    
    assert np.array_equal(img_no_mask, img_masked)
    assert not np.sum(img_masked) == 0