#Test main tree module, only run comet experiment locally to debug callbacks
import os
import pandas as pd
import glob

is_travis = 'TRAVIS' in os.environ
if not is_travis:
    from comet_ml import Experiment 
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")
    experiment.add_tag("testing") 
else:
    experiment = None

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
    train_config["epochs"] = 2
    train_config["steps"] = 2
    train_config["gpus"] = 1
    train_config["crop_size"] = 100
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
    mod.RGB_size = mod.config["train"]["RGB"]["crop_size"]
    mod.HSI_size = mod.config["train"]["HSI"]["crop_size"]
    mod.HSI_channels = 3
    mod.RGB_channels = 3
    mod.extend_box = mod.config["train"]["extend_box"]
    mod.classes_file = None
    mod.classes = mod.config["train"]["classes"]
    
    #Create a model with input sizes
    mod.create()
            
    return mod

@pytest.fixture()
def tfrecords(mod, tmpdir):
    created_records = mod.generate(shapefile=test_predictions, site=0, elevation=100, HSI_sensor_path=test_sensor_tile, RGB_sensor_path=test_sensor_tile, train=True, chunk_size=100)    
    return created_records

def test_generate(mod):
    created_records = mod.generate(shapefile=test_predictions, site=0, elevation=100, HSI_sensor_path=test_sensor_tile, RGB_sensor_path=test_sensor_tile, train=True, chunk_size=100)  
    assert all([os.path.exists(x) for x in created_records])

def test_split_data(mod, tfrecords):
    #Create class
    mod.read_data(validation_split=True)
    
    assert len(mod.train_split_records) > 0
    assert len(mod.test_split_records) > 0
    
    #Assert tfrecords are split
    assert all([x not in mod.train_split_records for x in mod.test_split_records])
    
@pytest.mark.parametrize("submodel",["spectral","spatial","metadata","None"])
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

#Test that the composition of the validation split is the same no matter the data
def test_read_data(mod, tfrecords):
    mod.read_data(mode="RGB_submodel", validation_split=True)    
    before = mod.test_split_records
    mod.read_data(mode="ensemble",validation_split=True)   
    after = mod.test_split_records
    assert before == after
    
def test_train(tfrecords, mod):
    #initial weights
    initial_weight = mod.RGB_model.layers[1].get_weights()
    
    mod.read_data(mode="RGB_train")
    mod.train(sensor="RGB")
    
    final_weight = mod.RGB_model.layers[1].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)

    assert "loss" in list(mod.RGB_model.history.history.keys())     
 
def test_ensemble(tfrecords, mod):    
    mod.read_data("ensemble",validation_split=True)
    mod.ensemble(experiment=None, class_weight=None)
     
@pytest.mark.skipif(is_travis, reason="Cannot load comet on TRAVIS")
def test_train_callbacks(tfrecords, mod):
    mod.classes_file = "{}/species_class_labels.csv".format(os.path.dirname(tfrecords[0]))
    
    
    with experiment.context_manager("metadata`"):
        mod.ensemble(experiment=experiment)     
    
    mod.read_data(validation_split=True, mode="RGB_submodel")
    
    with experiment.context_manager("ensemble"):
        mod.ensemble(experiment=experiment) 
    
    mod.read_data(validation_split=True, mode="RGB_submodel")
    mod.train(sensor="RGB", submodel="spectral",experiment=experiment)

    mod.read_data(validation_split=True, mode="RGB_train")
    mod.train(experiment=experiment, sensor="RGB")
    
   
    #assert experiment.get_metric("Within-site Error") > 0
    
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

    