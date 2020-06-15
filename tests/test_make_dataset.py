#test make_dataset

from DeepTreeAttention.generators import make_dataset

import tensorflow as tf
import numpy as np
import rasterio
import os
import pytest

@pytest.fixture()
def training_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    #Create a raster that looks data
    arr = np.random.rand(4, 1202,4738).astype(np.float)
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[0], width = arr.shape[1],
                                count=4, dtype=str(arr.dtype),
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, 4768, 1202))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn


@pytest.fixture()
def ground_truth_raster(tmp_path):
    fn = os.path.join(tmp_path,"ground_truth.tif")
    #Create a raster that looks data
    arr = np.random.randint(20,size=(1, 1202,4738)).astype(np.uint8)
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[0], width = arr.shape[1],
                                count=1, dtype=str(arr.dtype),
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, 4768, 1202))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

def test_tf_data_generator(training_raster, ground_truth_raster):
    iterator = make_dataset.tf_data_generator(training_raster, ground_truth_raster,crop_height=11,crop_width=11, sensor_channels=4)
    
    i = 0
    for data, labels in iterator:
        assert data.shape == (11,11,4)
        assert labels.shape == (1,)
        i+=1
        if i >=5: break 
    
#def test_training_dataset(training_raster, ground_truth_raster):
    
    ##assert exist
    #assert os.path.exists(training_raster)
    #assert os.path.exists(ground_truth_raster)
    
    #dataset = make_dataset.training_dataset(
        #sensor_list=[training_raster],
        #ground_truth_list=[ground_truth_raster],
        #crop_height=11,
        #crop_width=11,
        #batch_size=1
        #)
    
    #batch = dataset.take(1)
    #batch.shape == [1,1,1,1]
    
    
    
    