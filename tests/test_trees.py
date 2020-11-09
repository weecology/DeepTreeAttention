#Test main tree module, only run comet experiment locally to debug callbacks
import glob
import geopandas as gpd
import os

is_travis = 'TRAVIS' in os.environ
if not is_travis:
    from comet_ml import Experiment 
    experiment = Experiment(project_name="neontrees", workspace="bw4sz")
    experiment.add_tag("testing") 
else:
    experiment = None

import numpy as np
import pytest
import pandas as pd
import rasterio
import tensorflow as tf

from DeepTreeAttention.utils import metrics
from DeepTreeAttention import trees
from DeepTreeAttention.generators import boxes
from matplotlib.pyplot import imshow
from tensorflow.keras import metrics as keras_metrics

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

@pytest.fixture()
def mod(tmpdir):
    mod = trees.AttentionModel(config="conf/tree_config.yml")   
    
    
    train_dir = tmpdir.mkdir("train")
    predict_dir = tmpdir.mkdir("predict")
    label_file = "{}/label_file.csv".format(train_dir)
    
    #create a fake label file
    pd.DataFrame({"taxonID":["Ben","Jon"],"label":[0,1]}).to_csv(label_file)
    
    config = {}
    train_config = { }
    train_config["tfrecords"] = train_dir
    train_config["batch_size"] = 2
    train_config["epochs"] = 1
    train_config["steps"] = 1
    train_config["gpus"] = 1
    train_config["crop_size"] = 100
    train_config["shuffle"] = True
    train_config["weighted_sum"] = False
    train_config["classes"] = 2
    train_config["species_class_file"] = label_file
        
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
    mod.extend_HSI_box = mod.config["train"]["HSI"]["extend_box"]
    mod.classes_file = label_file
    #Create a model with input sizes
    mod.create()
            
    return mod

@pytest.fixture()
def tfrecords(mod, tmpdir):
    shp = gpd.read_file(test_predictions)
    
    created_records = mod.generate(shapefile=test_predictions, site=0, elevation=100,
                                   heights=np.random.random(shp.shape[0]),
                                   HSI_sensor_path=test_sensor_tile,
                                   RGB_sensor_path=test_sensor_tile,
                                   train=True,
                                   chunk_size=2)    
    return created_records

def test_generate(mod):
    shp = gpd.read_file(test_predictions)    
    created_records = mod.generate(
        shapefile=test_predictions,
        site=0,
        heights=np.random.random(shp.shape[0]),
        elevation=100,
        HSI_sensor_path=test_sensor_tile,
        RGB_sensor_path=test_sensor_tile,
        train=True, chunk_size=2)  
    
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
    
    shp = gpd.read_file(test_predictions)
    
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
    
    assert shp.shape[0] == train_counter + test_counter
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
    
def test_train_metadata(tfrecords, mod):
    #initial weights
    initial_weight = mod.metadata_model.layers[3].get_weights()
    
    mod.read_data(mode="metadata", validation_split=True)
    mod.train(submodel="metadata", experiment=experiment, class_weight=None)
    
    final_weight = mod.metadata_model.layers[3].get_weights()
    
    #assert training took place
    assert not np.array_equal(final_weight,initial_weight)
    assert "loss" in mod.metadata_model.history.history  
 
def test_ensemble(tfrecords, mod):    
    mod.read_data("ensemble",validation_split=True)
    mod.config["train"]["ensemble"]["epochs"] = 1   
    mod.config["train"]["ensemble"]["batch_size"] = 2
    mod.ensemble(experiment=experiment, class_weight=None, train=True)
         
@pytest.mark.skipif(is_travis, reason="Cannot load comet on TRAVIS")
def test_train_callbacks(tfrecords, mod):
    mod.read_data(validation_split=True, mode="RGB_submodel")
    
    #update epoch manually
    mod.config["train"]["RGB"]["epochs"] = 1
    mod.train(sensor="RGB", submodel="spectral",experiment=experiment)

    mod.read_data(validation_split=True, mode="RGB_train")
    mod.train(experiment=experiment, sensor="RGB")
        
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

    