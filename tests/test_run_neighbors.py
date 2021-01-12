#run_neighbors test
#
import geopandas as gpd
import os
import tensorflow as tf
import numpy as np
import pytest
import pandas as pd

from DeepTreeAttention.models import neighbors_model
from DeepTreeAttention.callbacks import callbacks
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.models import Hang2020_geographic as Hang
from DeepTreeAttention.models import metadata

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_sensor_hyperspec = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

@pytest.fixture()
def mod(tmpdir):
    mod = AttentionModel(config="conf/tree_config.yml")   
    
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
    eval_config["steps"] = 1
    eval_config["ground_truth_path"] = "data/processed/test.shp"
    
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
    mod.HSI_channels = 369
    mod.RGB_channels = 3
    mod.extend_HSI_box = mod.config["train"]["HSI"]["extend_box"]
    mod.classes_file = label_file
    mod.train_shp = pd.DataFrame({"taxonID":["Jon","Ben"], "siteID":[0,1],"domainID":[0,1],"plotID":[0,1], "canopyPosition":["a","b"],"scientific":["genus species","genus species"]})
    mod.train_shp.index =[2,7]
    mod.sites = 23
    mod.domains = 15
    
    #Create a model with input sizes
    mod.create()
    return mod

@pytest.fixture()
def ensemble_model(mod):
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=mod.HSI_size, width=mod.HSI_size, channels=mod.HSI_channels)    
    model1 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    
    metadata_model = metadata.create(classes=2, sites=23, domains=15, learning_rate=0.001)
    ensemble = Hang.learned_ensemble(HSI_model=model1, metadata_model=metadata_model, classes=2)
    ensemble = tf.keras.Model(ensemble.inputs, ensemble.get_layer("submodel_concat").output)
    
    return ensemble
    
@pytest.fixture()
def tfrecords(mod, ensemble_model, tmpdir):    
    created_records = mod.generate(shapefile=test_predictions, site=0, domain=1, elevation=100,
                                   HSI_sensor_path=test_sensor_hyperspec,
                                   RGB_sensor_path=test_sensor_tile,
                                   train=True,
                                   chunk_size=2,
                                   savedir = mod.config["train"]["tfrecords"],
                                   raw_boxes=test_predictions,
                                   ensemble_model=ensemble_model
                                   )    
    return created_records

def test_run_neighbors(mod, tfrecords):
    #Create a class and run
    mod.ensemble(experiment=None, train=False)
    
    #turn ensemble model into a feature extractor of the 2nd to last layer.
    mod.ensemble_model = tf.keras.Model(mod.ensemble_model.inputs, mod.ensemble_model.get_layer("submodel_concat").output)    
       
    mod.read_data("neighbors")
    neighbor = neighbors_model.create(ensemble_model = mod.ensemble_model, k_neighbors=mod.config["neighbors"]["k_neighbors"], classes=mod.classes)
    labeldf = pd.read_csv(mod.classes_file)
    label_names = list(labeldf.taxonID.values)
    
    for data, labels in mod.train_split:
        HSI_image, elevation, site, domains, neighbor_array  = data
        assert HSI_image.shape == (2,20,20,369)
        assert elevation.shape == (2)
        assert site.shape == (2,mod.sites)
        assert domains.shape == (2, mod.domains)
        assert neighbor_array.shape == (2,mod.config["neighbors"]["k_neighbors"], mod.ensemble_model.output.shape[1])
        assert labels.shape == (2, mod.classes)
        
    neighbor.fit(
        mod.train_split,
        epochs=mod.config["train"]["ensemble"]["epochs"])    
    