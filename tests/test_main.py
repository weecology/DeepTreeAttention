#Test main
import pytest
import os
import rasterio
import numpy as np
from DeepTreeAttention import main
from DeepTreeAttention.generators import make_dataset

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
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, arr.shape[1], arr.shape[2]))
    
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
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, arr.shape[1], arr.shape[2]))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def tfrecords(training_raster, ground_truth_raster,tmpdir):
    tfrecords = make_dataset.generate(training_raster, ground_truth_raster,savedir=tmpdir)
    
    return os.path.dirname(tfrecords[0])

@pytest.fixture()
def test_config(tfrecords):
    config = {}
    train_config = { }
    train_config["tfrecords"] = tfrecords
    train_config["batch_size"] = 10
    train_config["epochs"] = 1
    train_config["steps"] = 2
    train_config["crop_size"] = 11
    train_config["sensor_channels"] = 4
    train_config["shuffle"] = False
        
    #evaluation
    eval_config = { }
    eval_config["tfrecords"] = tfrecords
    
    config["train"] = train_config
    config["evaluation"] = eval_config
    
    return config
    
def test_AttentionModel(test_config):

    #Create class
    mod = main.AttentionModel()      
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
        
    #Create model
    mod.create()
    mod.read_data()
    
    mod.config["evaluation"]["sensor_path"] = None
    mod.config["evaluation"]["ground_truth_path"] = None
    
    #initial weights
    initial_weight = mod.model.layers[1].get_weights()
    
    #train
    mod.train()
    
    final_weight = mod.model.layers[1].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)
    
def test_predict(test_config):
    #Create class
    mod = main.AttentionModel()    
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Create
    mod.create()
    mod.read_data()
    
    result = mod.model.predict(mod.testing_set, steps=1)
    
    assert result.shape == (mod.config["train"]["batch_size"], mod.config["train"]["classes"])

def test_evaluate(test_config):
    #Create class
    mod = main.AttentionModel()    
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Create
    mod.create()
    mod.read_data()
        
    result = mod.evaluate(steps=2)
    print(result)
    
    
    