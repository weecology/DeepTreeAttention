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
    arr = np.random.rand(4, 25,25).astype(np.float)
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype=str(arr.dtype),
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, 4768, 1202))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn


@pytest.fixture()
def ground_truth_raster(tmp_path):
    fn = os.path.join(tmp_path,"ground_truth.tif")
    #Create a raster that looks data (smaller)
    arr = np.random.randint(20,size=(1, 50,50)).astype(np.uint8)
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype=str(arr.dtype),
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, 4768, 1202))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

def test_get_coordinates(ground_truth_raster):
    results = make_dataset.get_coordinates(ground_truth_raster)
    
    #assert every pixel has a label
    assert results.shape[0] == 50 * 50

def test_select_crops(ground_truth_raster, training_raster):
    
    results = make_dataset.get_coordinates(ground_truth_raster)
    coordinates = zip(results.easting, results.northing)
    crops = make_dataset.select_crops(training_raster, coordinates, size=5)
    
    assert all([x.shape == (5,5,4) for x in crops])
    
def test_tf_data_generator(training_raster, ground_truth_raster):
    #Tensorflow encodes string as b bytes
    iterator = make_dataset.tf_data_generator(training_raster.encode(), ground_truth_raster.encode(),size=11, classes=20)
    
    i = 0
    for data, labels in iterator:
        assert data.shape == (11,11,4)
        assert labels.shape == (20,)
        i+=1
        if i >=5: break 
    
def test_tf_dataset(training_raster, ground_truth_raster):

    dataset = make_dataset.tf_dataset(
        sensor_path=training_raster.encode(),
        ground_truth_path=ground_truth_raster.encode(),
        size=11,
        batch_size=10,
        classes=20
        )
    
    for data, label in dataset.take(2):  # only take first element of dataset
        numpy_data = data.numpy()
        numpy_labels = label.numpy()
        
    assert numpy_data.shape == (10,11,11,4)
    assert numpy_labels.shape == (10,20)
    
    
    
    