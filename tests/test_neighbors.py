#Test Neighbors module
import pytest
import glob
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import rasterio
import tensorflow.keras as tfk
import tensorflow as tf

from DeepTreeAttention.generators import neighbors
from DeepTreeAttention.models.Hang2020_geographic import create_models, learned_ensemble
from DeepTreeAttention.models.metadata import create as create_metadata
from DeepTreeAttention.trees import AttentionModel


##Global variables
#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_hsi_tile = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

site_classes_file =  "data/processed/site_class_labels.csv"
site_classdf  = pd.read_csv(site_classes_file)
site_label_dict = site_classdf.set_index("siteID").label.to_dict()

domain_classes_file =  "data/processed/domain_class_labels.csv"    
domain_classdf  = pd.read_csv(domain_classes_file)
domain_label_dict = domain_classdf.set_index("domainID").label.to_dict()

hyperspectral_pool = ['data/raw/2019_BART_5_320000_4881000_image_crop.tif']    

@pytest.fixture()
def data():
    data = gpd.read_file(test_predictions)
    
    return data

@pytest.fixture()
def metadata(mod):
    #encode metadata
    site = "BART"
    numeric_site = site_label_dict[site]
    one_hot_sites = tf.one_hot(numeric_site, mod.sites)
    
    domain = "D17"
    numeric_domain = domain_label_dict[domain]   
    one_hot_domains = tf.one_hot(numeric_domain, mod.domains)
    
    #ToDO bring h5 into here.
    #elevation = elevation_from_tile(sensor_path)/1000
    elevation = 100/1000
    metadata = [elevation, one_hot_sites, one_hot_domains]
    
    return metadata

@pytest.fixture()
def df():
    df = gpd.read_file(test_predictions)
    df["individual"] = np.arange(df.shape[0])
    df["siteID"] = "BART"
    df["domainID"] = "D17"
    
    return df

@pytest.fixture()
def mod(tmpdir):
    mod = AttentionModel(config="conf/tree_config.yml")   
    mod.sites = 2
    mod.domains = 2    
    mod.RGB_channels = 3
    mod.HSI_channels = 3
    
    train_dir = tmpdir.mkdir("train")
    
    label_file = "{}/label_file.csv".format(train_dir)
    
    shp = gpd.read_file(test_predictions)
    mod.config["train"]["tfrecords"] = train_dir
    mod.classes_file = "data/processed/species_class_labels.csv"
    created_records = mod.generate(shapefile=test_predictions, site=0, domain=1, elevation=100,
                                   heights=np.random.random(shp.shape[0]),
                                   HSI_sensor_path=test_sensor_tile,
                                   RGB_sensor_path=test_sensor_tile,
                                   train=True,
                                   chunk_size=2)  
    
    #create a fake label file
    pd.DataFrame({"taxonID":["Ben","Jon"],"label":[0,1]}).to_csv(label_file)
    mod.classes_file = label_file
    
    mod.create()
    mod.ensemble(experiment=None, train=False)
    
    #turn ensemble model into a feature extractor of the 2nd to last layer.
    mod.ensemble_model = tfk.Model(mod.ensemble_model.inputs, mod.ensemble_model.get_layer("submodel_concat").output)
    
    return mod

def test_predict_neighbors(data, metadata, mod):
    target = data.iloc[0]
    neighbor_pool = data[~(data.index == target.index)]
    raster = rasterio.open(test_sensor_tile)
    feature_array, distances = neighbors.predict_neighbors(target, metadata=metadata, HSI_size=20, raster=raster, neighbor_pool=neighbor_pool, model=mod.ensemble_model, k_neighbors=5)
    assert feature_array.shape[0] == 5
    assert feature_array.shape[1] == mod.ensemble_model.get_layer("submodel_concat").output.shape[1]

    assert len(distances) == 5
    
def test_extract_features(mod, df, tmpdir):
    x = df.individual.values[0]
    feature_array, distances = neighbors.extract_features(df=df, x=x, model_class=mod, hyperspectral_pool=hyperspectral_pool, site_label_dict=site_label_dict, domain_label_dict=domain_label_dict, k_neighbors=5)
    assert feature_array.shape[0] == 5
    assert feature_array.shape[1] == mod.ensemble_model.get_layer("submodel_concat").output.shape[1]    
    
    assert len(distances) == 5
    
def test_predict_dataframe(mod, df):
    results_dict = neighbors.predict_dataframe(df=df, model_class =  mod, hyperspectral_pool=hyperspectral_pool, site_label_dict=site_label_dict, domain_label_dict=domain_label_dict, k_neighbors=5)
    assert len(results_dict) == df.shape[0]

def test_predict_dataframe_with_padding(mod, df):
    #manipulate the input data to less than k_neighbors, the algorithm should pad with zeros.
    df = df.head(3)
    results_dict = neighbors.predict_dataframe(df=df, model_class=mod, hyperspectral_pool=hyperspectral_pool, site_label_dict=site_label_dict, domain_label_dict=domain_label_dict)
    
    assert len(results_dict) == df.shape[0]
    
    assert len(results_dict[0][0]) == 5
    assert len(results_dict[0][1]) == 5
