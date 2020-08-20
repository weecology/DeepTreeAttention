#Test main tree module, only run comet experiment locally to debug callbacks
import os
import glob

is_travis = 'TRAVIS' in os.environ
if not is_travis:
    from comet_ml import Experiment 
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")

import pytest
import rasterio
import numpy as np
import tensorflow as tf

from DeepTreeAttention.utils import metrics
from DeepTreeAttention import trees
from DeepTreeAttention.generators import boxes
from matplotlib.pyplot import imshow
from tensorflow.keras import metrics as keras_metrics

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_edited.shp"

test_field_data = "data/processed/field_data_1.tfrecord"
#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

@pytest.fixture()
def mod(tmpdir):
    mod = trees.AttentionModel(config="conf/tree_config.yml")   
    
    train_dir = tmpdir.mkdir("train")
    predict_dir = tmpdir.mkdir("predict")
    
    config = {}
    train_config = { }
    train_config["tfrecords"] = train_dir
    train_config["batch_size"] = 32
    train_config["epochs"] = 1
    train_config["steps"] = 2
    train_config["gpu"] = 1
    train_config["crop_size"] = 100
    train_config["sensor_channels"] = 3
    train_config["shuffle"] = True
    train_config["weighted_sum"] = False
    train_config["classes"] = 2
        
    #evaluation
    eval_config = { }
    eval_config["tfrecords"] = None
    eval_config["steps"] = 2
    
    predict_config = { }
    predict_config["tfrecords"] = predict_dir
        
    config["train"] = train_config
    config["evaluation"] = eval_config
    config["predict"] = predict_config
    
    #Replace config for testing env
    for key, value in config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Update the inits
    mod.height = mod.config["train"]["crop_size"]
    mod.width = mod.config["train"]["crop_size"]
    mod.channels = mod.config["train"]["sensor_channels"]
    mod.weighted_sum = mod.config["train"]["weighted_sum"]
    mod.extend_box = mod.config["train"]["extend_box"]
    mod.classes_file = os.path.join(mod.config["train"]["tfrecords"],"class_labels.csv")
    mod.classes = mod.config["train"]["classes"]
    
    #Create a model with input sizes
    mod.create()
            
    return mod

@pytest.fixture()
def tfrecords(mod, tmpdir):
    created_records = mod.generate(shapefile=test_predictions, sensor_path=test_sensor_tile, train=True, chunk_size=100)    
    return created_records

def test_generate(mod):
    created_records = mod.generate(shapefile=test_predictions, sensor_path=test_sensor_tile, train=True, chunk_size=100)  
    assert all([os.path.exists(x) for x in created_records])

def test_split_data(mod, tfrecords):
    #Create class
    mod.read_data(validation_split=True)
    
    assert len(mod.train_split_records) > 0
    assert len(mod.test_split_records) > 0
    
    #Assert tfrecords are split
    assert all([x not in mod.train_split_records for x in mod.test_split_records])
    
@pytest.mark.parametrize("submodel",["spectral","spatial","None"])
def test_AttentionModel(mod, tfrecords, submodel):
    mod.read_data(validation_split=True)
    
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

def test_train(tfrecords, mod):
    #initial weights
    initial_weight = mod.model.layers[1].get_weights()
    
    mod.read_data()
    mod.train()
    
    final_weight = mod.model.layers[1].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)

    assert "loss" in list(mod.model.history.history.keys())     
         
def test_evaluate(mod, tfrecords):
    #Create
    mod.read_data(validation_split=True)
    
    metric_list = [
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

@pytest.mark.skipif(is_travis, reason="Cannot load comet on TRAVIS")
def test_train_callbacks(tfrecords, mod):
    mod.read_data(validation_split=True, mode="submodel")
    mod.train(experiment=experiment)

@pytest.mark.skipif(is_travis, reason="Cannot load comet on TRAVIS")
def test_train_field_callbacks(mod):
    mod.config["train"]["tfrecords"] = "data/processed/"
    mod.height = 20
    mod.width = 20
    mod.channels = 369
    mod.create()
    mod.read_data(validation_split=True)
    mod.classes_file = "data/processed/class_labels.csv"
    mod.train(experiment=experiment)

    