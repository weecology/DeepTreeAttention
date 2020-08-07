#Test Houston2018 main module
import pytest
import os
import rasterio
import numpy as np

from DeepTreeAttention import Houston2018
from DeepTreeAttention.generators import make_dataset
from matplotlib.pyplot import imshow
from DeepTreeAttention.visualization import visualize
from DeepTreeAttention.utils import metrics

import tensorflow as tf
from tensorflow.keras import metrics as keras_metrics

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
                                transform=rasterio.transform.from_origin(272056.0, 3289689.0, 1, -1))
    
    new_dataset.write(arr)
    new_dataset.close()

    return fn

@pytest.fixture()
def training_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    
    #Create a raster that looks data, index order to help id
    arr = np.arange(30 * 30).reshape((30,30))
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

@pytest.fixture()
def predict_raster(tmp_path):
    fn = os.path.join(tmp_path,"training.tif")
    
    #Create a raster that looks data, index order to help id
    arr = np.arange(12 * 15).reshape((12,15))
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

@pytest.fixture()
def tfrecords(training_raster, ground_truth_raster,tmpdir):
    tfrecords = make_dataset.generate_training(training_raster, ground_truth_raster, size=5, savedir=tmpdir,n_chunks=3)
    
    return os.path.dirname(tfrecords[0])

@pytest.fixture()
def predict_tfrecords(predict_raster,tmpdir):
    tfrecords = make_dataset.generate_raster_prediction(predict_raster, savedir=tmpdir, size=5,chunk_size=100)
    return tfrecords

@pytest.fixture()
def test_config(tfrecords):
    config = {}
    train_config = { }
    train_config["tfrecords"] = tfrecords
    train_config["batch_size"] = 32
    train_config["epochs"] = 1
    train_config["steps"] = 2
    train_config["gpu"] = 1
    train_config["crop_size"] = 5
    train_config["sensor_channels"] = 4
    train_config["shuffle"] = False
        
    #evaluation
    eval_config = { }
    eval_config["tfrecords"] = tfrecords
    eval_config["steps"] = 2
    
    config["train"] = train_config
    config["evaluation"] = eval_config
    
    return config

@pytest.mark.parametrize("weighted_sum",[False, True])
def test_AttentionModel(test_config,weighted_sum):

    #Create class
    mod = Houston2018.AttentionModel(config="conf/houston_config.yml")
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    mod.config["train"]["weighted_sum"] = weighted_sum
        
    #Create model
    mod.create()
    mod.read_data(validation_split=True)
        
    #initial weights
    initial_weight = mod.model.layers[1].get_weights()
    
    #How many batches and ensure no overlap in data
    train_image_data = []
    test_image_data = []
    
    train_counter=0
    for data, label in mod.train_split:
        train_image_data.append(data)
        train_counter+=data.shape[0]
            
    test_counter=0
    for data, label in mod.val_split:
        test_image_data.append(data)            
        test_counter+=data.shape[0]
    
    assert train_counter > test_counter
    
    #No test in train batches
    assert all([not np.array_equal(y,x) for x in train_image_data for y in test_image_data])
    
    #Spatial block of train and test
    test_center_pixel = test_image_data[0][0,2,2,0].numpy()
    test_pixel_image = test_image_data[0][0,:,:,0].numpy()        
    for x in train_image_data:    
        assert not test_center_pixel in x[:,2,2,0]
    
    #train
    mod.train()
    
    final_weight = mod.model.layers[1].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)

    assert "val_acc" in list(mod.model.history.history.keys()) 
        
def test_predict_raster(test_config, predict_tfrecords):
    #Create class
    mod = Houston2018.AttentionModel(config="conf/houston_config.yml")
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Create
    mod.create()
    mod.read_data()
    results = mod.predict_raster(predict_tfrecords, batch_size=2)
    predicted_raster = visualize.create_raster(results)
    
    #Equals size of the input raster
    assert predicted_raster.shape == (12,15)
    
def test_evaluate(test_config):
    #Create class
    mod = Houston2018.AttentionModel(config="conf/houston_config.yml")    
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Create
    mod.create()
    mod.read_data(validation_split=True)
    
    metric_list = [
        keras_metrics.TopKCategoricalAccuracy(k=2, name="top_k"),
        keras_metrics.CategoricalAccuracy(name="acc")
    ]
    
    mod.model.compile(loss="categorical_crossentropy",
                               optimizer=tf.keras.optimizers.Adam(
                                   lr=float(mod.config['train']['learning_rate'])),
                               metrics=metric_list)
    
    #Method 1, class eval method
    print("Before evaluation")
    y_pred, y_true = mod.evaluate(mod.val_split)
    
    print("evaluated")
    test_acc = keras_metrics.CategoricalAccuracy()
    test_acc.update_state(y_true=y_true, y_pred = y_pred)
    method1_eval_accuracy = test_acc.result().numpy()
    
    assert y_pred.shape == y_true.shape

    #Method 2, keras eval method
    metric_list = mod.model.evaluate(mod.val_split)
    metric_dict = {}
    for index, value in enumerate(metric_list):
        metric_dict[mod.model.metrics_names[index]] = value
    
    assert method1_eval_accuracy == metric_dict["acc"]   
    
    #F1 requires integer, not softmax
    f1s = metrics.f1_scores( y_true, y_pred)    
    
#Submodel compile and train
@pytest.mark.parametrize("submodel",["spectral","spatial"])
def test_submodel_AttentionModel(test_config, submodel):
    #Create class
    mod = Houston2018.AttentionModel(config="conf/houston_config.yml")
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
        
    #Create model
    mod.create()
    mod.read_data(mode="submodel")
    mod.train(submodel=submodel)
    
    for batch in mod.train_split:
        len(batch)
        
def test_split_data(test_config):
    #Create class
    mod = Houston2018.AttentionModel(config="conf/houston_config.yml")   
    
    #Replace config for testing env
    for key, value in test_config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
            
    mod.read_data(validation_split=True)
    
    #Assert tfrecords are split
    assert all([x not in mod.train_split_records for x in mod.test_split_records])