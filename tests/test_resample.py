#test resample utils
import rasterio
import numpy as np
import os
import pytest

from DeepTreeAttention.utils import resample

@pytest.fixture()
def ground_truth_raster(tmp_path):
    fn = os.path.join(tmp_path,"ground_truth.tif")
    #Create a raster that looks data (smaller)
    arr = np.random.randint(1,21,size=(1, 30,30)).astype(np.uint16)
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_origin(272056.0, 3289689.0, 0.5, -0.5))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def training_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    
    #Create a raster that looks data, index order to help id
    arr = np.arange(15 * 15).reshape((15,15))
    arr = np.dstack([arr]*4)
    arr = np.rollaxis(arr, 2,0)
    arr = arr.astype("uint16")
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_origin(272056.0, 3289689.0, 1, -1))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

def test_create_tif(ground_truth_raster, tmp_path):
    src = rasterio.open(ground_truth_raster)
    data = src.read()
    
    filename = "{}/test.tif".format(tmp_path)
    resample.create_tif(source_tif=ground_truth_raster, filename=filename, numpy_array=data)
    
    #Assert the original and saved are identical
    saved_src = rasterio.open(filename)
    saved_data = saved_src.read()
    
    assert saved_src.shape == src.shape
    np.testing.assert_array_almost_equal(data, saved_data) 
    
def test_resample(ground_truth_raster, training_raster, tmp_path):
    """Assert that resampled training raster has the shape of the ground truth raster"""
    src = rasterio.open(training_raster)
    data = src.read(1)
    data = np.expand_dims(data,0)
    filename = "{}/test.tif".format(tmp_path)
    resample.create_tif(source_tif=training_raster, filename=filename, numpy_array=data)    
    
    resampled_filename = resample.resample(filename)
    resampled_src = rasterio.open(resampled_filename)
    ground_truth_raster_src = rasterio.open(ground_truth_raster)
    
    resampled_data = resampled_src.read()
    assert ground_truth_raster_src.shape == resampled_src.shape
    assert ground_truth_raster_src.bounds == resampled_src.bounds
    