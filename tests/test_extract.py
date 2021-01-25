#Test Extract
from DeepTreeAttention import trees
from DeepTreeAttention.visualization import extract
import pandas as pd
import os
import glob
import pytest

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_sensor_hyperspec = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

@pytest.fixture()
def mod(tmpdir):
    mod = trees.AttentionModel(config="conf/tree_config.yml")   
    
    train_dir = tmpdir.mkdir("train")
    label_file = "{}/label_file.csv".format(train_dir)
    
    #create a fake label file
    pd.DataFrame({"taxonID":["Ben","Jon"],"label":[0,1]}).to_csv(label_file)
    
    config = {}
    train_config = { }
    train_config["tfrecords"] = train_dir
    train_config["batch_size"] = 1
    train_config["epochs"] = 1
    train_config["steps"] = 1
    train_config["gpus"] = 1
    train_config["crop_size"] = 20
    train_config["shuffle"] = True
    train_config["weighted_sum"] = True
  
    config["train"] = train_config
    
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
def tfrecords(mod, tmpdir):    
    
    created_records = mod.generate(shapefile=test_predictions, site=0, domain=1, elevation=100,
                                   HSI_sensor_path=test_sensor_hyperspec,
                                   RGB_sensor_path=test_sensor_tile,
                                   train=True,
                                   chunk_size=2,
                                   savedir = mod.config["train"]["tfrecords"]
                                   )    
    return created_records

def test_save_images_to_matlab(mod,tfrecords, tmpdir):
    
    #For testing purposes, split a val set
    mod.config["evaluation"]["tfrecords"] =  mod.config["train"]["tfrecords"]
    extract.save_images_to_matlab(DeepTreeAttention=mod, classes=["Ben"], savedir=tmpdir)
    
    printed_images = glob.glob("{}/*.mat".format(tmpdir))
    
    assert len(printed_images) == 9
    
    
    
    