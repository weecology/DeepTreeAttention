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
    arr = np.random.rand(4, 25,25).astype("uint16")
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, arr.shape[1], arr.shape[2]))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def predict_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    #Create a raster that looks data
    arr = np.random.rand(4, 10,10).astype(np.float)
    
    #hard coded from Houston 2018 ground truth
    new_dataset = rasterio.open(fn, 'w', driver='GTiff',
                                height = arr.shape[1], width = arr.shape[2],
                                count=arr.shape[0], dtype="uint16",
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
                                count=arr.shape[0], dtype="uint16",
                                crs=rasterio.crs.CRS.from_epsg("26915"),
                                transform=rasterio.transform.from_bounds(272056.0, 3289689.0, 274440.0, 3290290.0, arr.shape[1], arr.shape[2]))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def tfrecords(training_raster, ground_truth_raster,tmpdir):
    tfrecords = make_dataset.generate_training(training_raster, ground_truth_raster,savedir=tmpdir)
    
    return os.path.dirname(tfrecords[0])

@pytest.fixture()
def predict_tfrecords(predict_raster,tmpdir):
    tfrecords = make_dataset.generate_prediction(predict_raster, savedir=tmpdir, chunk_size=100)
    return tfrecords

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

@pytest.mark.parametrize("validation_split",[False, True])
def test_AttentionModel(test_config,validation_split):

    #Create class
    mod = main.AttentionModel()      
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
        
    #Create model
    mod.create()
    mod.read_data(validation_split=validation_split)
    
    mod.config["evaluation"]["sensor_path"] = None
    mod.config["evaluation"]["ground_truth_path"] = None
    
    #initial weights
    initial_weight = mod.model.layers[1].get_weights()
    
    #train
    mod.train()
    
    final_weight = mod.model.layers[1].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)
    
    #assert val acc exists if split
    if validation_split:
        assert "val_acc" in list(mod.model.history.history.keys()) 
        
def test_predict(test_config, predict_tfrecords):
    #Create class
    mod = main.AttentionModel()    
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Create
    mod.create()
    mod.read_data()
    predicted_raster = mod.predict(predict_tfrecords)
    
    #Equals size of the input raster
    assert predicted_raster.shape == (10,10)

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
        
    result = mod.evaluate(mod.train_split, steps=2)
    assert "acc" in list(result.keys())
    print(result)
    
    
    