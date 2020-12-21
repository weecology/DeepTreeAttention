#Test Neighbors module
import pytest
import geopandas as gpd
import numpy as np
import rasterio
import tensorflow.keras as tfk

from DeepTreeAttention.generators import neighbors
from DeepTreeAttention.models.Hang2020_geographic import create_models, learned_ensemble
from DeepTreeAttention.models.metadata import create as create_metadata

#random label predictions just for testing
test_predictions = "data/raw/2019_BART_5_320000_4881000_image_small.shp"

#Use a small rgb crop as a example tile
test_sensor_tile = "data/raw/2019_BART_5_320000_4881000_image_crop.tif"

test_hsi_tile = "data/raw/2019_BART_5_320000_4881000_image_hyperspectral_crop.tif"

@pytest.fixture()
def data():
    data = gpd.read_file(test_predictions)
    
    return data

@pytest.fixture()
def model():
    HSI_model, _, _= create_models(height=20, width=20, channels=3, classes=2, learning_rate=0.001, weighted_sum=True)
    metadata_model = create_metadata(classes=2, sites=2, domains=2, learning_rate=0.01)
    ensemble_model = learned_ensemble(HSI_model=HSI_model, metadata_model=metadata_model, classes=2)
    feature_extractor = tfk.Model(ensemble_model.inputs, ensemble_model.get_layer("submodel_concat").output)
    
    return feature_extractor

@pytest.fixture()
def metadata():
    #encode metadata
    
    elevation = np.expand_dims(0.1,axis=0)
    site = np.expand_dims([0,1],axis=0)
    domain = np.expand_dims([0,1],axis=0)
    
    return [elevation, site, domain]
    
def test_predict_neighbors(data, metadata, model):
    target = data.iloc[0]
    neighbor_pool = data[~(data.index == target.index)]
    raster = rasterio.open(test_sensor_tile)
    feature_array = neighbors.predict_neighbors(target, metadata=metadata, HSI_size= 20, raster = raster, neighbor_pool = neighbor_pool, model=model,n=5)
    assert feature_array.shape[0] == 5
    assert feature_array.shape[1] == model.get_layer("submodel_concat").output.shape[1]

    