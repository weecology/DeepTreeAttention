#test make_dataset
from DeepTreeAttention.generators import make_dataset, create_tfrecords
from distributed import Client
import tensorflow as tf
import numpy as np
import rasterio
import os
import pytest
import pandas as pd

@pytest.fixture()
def training_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    
    #Create a raster that looks data, index order to help id
    arr = np.arange(25 * 25).reshape((25,25))
    arr = np.dstack([arr]*4)
    arr = np.rollaxis(arr, 2,0)
    arr = arr.astype("uint16")
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_origin(272056.0, 3289689.0, 1, 1))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn


@pytest.fixture()
def ground_truth_raster(tmp_path):
    fn = os.path.join(tmp_path,"ground_truth.tif")
    #Create a raster that looks data (smaller)
    arr = np.random.randint(20,size=(1, 50,50)).astype(np.uint16)
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_origin(272056.0, 3289689.0, 0.5, 0.5))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def train_tfrecords(training_raster, ground_truth_raster,tmpdir):
    client = Client()
    tfrecords = make_dataset.generate_training(training_raster, ground_truth_raster,savedir=tmpdir, chunk_size=100, use_dask=True, client=client)
    return tfrecords

@pytest.fixture()
def predict_tfrecords(training_raster,tmpdir):
    tfrecords = make_dataset.generate_prediction(sensor_path=training_raster, savedir=tmpdir,chunk_size=100)
    return tfrecords

def test_get_coordinates(ground_truth_raster):
    results = make_dataset.get_coordinates(ground_truth_raster)
    
    #assert every pixel has a label
    assert results.shape[0] == 50 * 50
    
    src = rasterio.open(ground_truth_raster)
    A = src.read()
    
    results.iloc[0].label == A[0,0,0]
    results.iloc[1].label == A[0,1,0]

def test_select_training_crops(ground_truth_raster, training_raster):
    results = make_dataset.get_coordinates(ground_truth_raster)
    coordinates = zip(results.easting, results.northing)
    crops, x, y = make_dataset.select_training_crops(training_raster, coordinates, size=5)
    
    assert all([x.shape == (5,5,4) for x in crops])
    assert all([x.sum()> 0 for x in crops])
    assert results.shape[0] == len(crops)
    
    #should be approxiately in the top corner, allow for pixel rounding. The sensor raster is filled with index values
    target = np.array( [ [0, 0, 0,0,0], [0, 0, 0,0,0] , [0,0,0,1,2] , [0,0,25,26,27],[0,0,50,51,52]])
    np.testing.assert_almost_equal(crops[0][:,:,0],target)
    
@pytest.mark.parametrize("use_dask",[False,True])
def test_generate_training(training_raster, ground_truth_raster,tmpdir, use_dask):
    if use_dask:
        client = Client()
    else:
        client = None
    tfrecords = make_dataset.generate_training(training_raster, ground_truth_raster, savedir=tmpdir, use_dask=use_dask, client=client)
    
    for path in tfrecords:
        assert os.path.exists(path)
        os.remove(path)

@pytest.mark.parametrize("use_dask",[False,True])
def test_generate_prediction(training_raster,tmpdir, use_dask):
    if use_dask:
        client = Client()
    else:
        client = None
    
    tfrecords = make_dataset.generate_prediction(training_raster, savedir=tmpdir, use_dask=use_dask, client=client)
        
    for path in tfrecords:
        assert os.path.exists(path)
        os.remove(path)
        
def test_tf_dataset_train(train_tfrecords):
    print(train_tfrecords)
    #Tensorflow encodes string as b bytes
    dataset = make_dataset.tf_dataset(train_tfrecords, batch_size=5, shuffle=False)
    
    counter=0
    for data, label in dataset:  # only take first element of dataset
        numpy_data = data.numpy()
        numpy_labels = label.numpy()
        counter+=data.shape[0]
    
    #one epoch should be every pixel in the raster
    assert counter == 50 * 50
    assert numpy_data.shape == (5,11,11,4)
    assert numpy_labels.shape == (5,20)
    
def test_tf_dataset_predict(predict_tfrecords):
    #Tensorflow encodes string as b bytes
    dataset = make_dataset.tf_dataset(predict_tfrecords, batch_size=10, shuffle=False, train=False)
    
    counter =0
    for data, x, y in dataset:  # only take first element of dataset
        numpy_data = data.numpy()
        x = x.numpy()
        y = y.numpy()
        counter+=data.shape[0]
    
    assert counter == 25 * 25
    