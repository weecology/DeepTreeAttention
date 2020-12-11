#Test cleaning
import glob
import geopandas as gpd
import os
import numpy as np
import pytest
import pandas as pd
import rasterio
import tensorflow as tf

from DeepTreeAttention import trees
from DeepTreeAttention.generators import boxes

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_sensor_hyperspec = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

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
    train_config["ground_truth_path"] = label_file    
        
    autoencoder_config = {}
    autoencoder_config["epochs"] = 1
    
    #evaluation
    eval_config = { }
    eval_config["tfrecords"] = None
    eval_config["steps"] = 1
    eval_config["ground_truth_path"] = test_predictions
    
    predict_config = { }
    predict_config["tfrecords"] = predict_dir
        
    config["train"] = train_config
    config["evaluation"] = eval_config
    config["predict"] = predict_config
    config["autoencoder"] = autoencoder_config
    
    #Replace config for testing env
    for key, value in config.items():
        for nested_key, nested_value in value.items():
            mod.config[key][nested_key] = nested_value
    
    #Update the inits
    mod.RGB_size = mod.config["train"]["RGB"]["crop_size"]
    mod.HSI_size = mod.config["train"]["HSI"]["crop_size"]
    mod.HSI_channels = 369
    mod.RGB_channels = 3
    mod.extend_HSI_box = mod.config["train"]["HSI"]["extend_box"]
    mod.classes_file = label_file
    mod.sites = 23
    mod.domains = 15
    shp = gpd.read_file(test_predictions)
    shp["id"] = shp.index.values    
    mod.train_shp = shp
    
    #Create a model with input sizes
    mod.create()
            
    return mod

@pytest.fixture()
def tfrecords(mod, tmpdir):
    shp = gpd.read_file(test_predictions)
    
    created_records = mod.generate(shapefile=test_predictions, site=0, domain=1, elevation=100,
                                   heights=np.random.random(shp.shape[0]),
                                   HSI_sensor_path=test_sensor_hyperspec,
                                   RGB_sensor_path=test_sensor_tile,
                                   train=True,
                                   chunk_size=2)    
    return created_records


def test_autoencoder_model(mod, tfrecords):
    mod.read_data("HSI_autoencoder")
    results = mod.find_outliers()
    assert not results.empty