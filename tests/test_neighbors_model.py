#test create model
from DeepTreeAttention.models import Hang2020_geographic as Hang
from DeepTreeAttention.models import metadata
from DeepTreeAttention.models import neighbors_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def metadata_data():
    #simulate data
    elevation = np.random.random(1)
    
    sites = np.zeros(10)
    sites[8] = 1
    sites = np.expand_dims(sites,0)
    
    domains = np.zeros(10)
    domains[8] = 1
    domains = np.expand_dims(domains,0)
    
    return [elevation, sites, domains]

@pytest.fixture()
def HSI_image():
    image = np.zeros((1,20, 20, 369), dtype=tf.keras.backend.floatx())
    return image

def test_neighbors(HSI_image, metadata_data):
    batch, height, width, channels = HSI_image.shape     
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=height, width=width, channels=channels)    
    model1 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    
    metadata_model = metadata.create(classes=2, sites=10, domains=10, learning_rate=0.001)
    ensemble = Hang.learned_ensemble(HSI_model=model1, metadata_model=metadata_model, classes=2)
    
    extractor = tf.keras.Model(ensemble.inputs,ensemble.get_layer("submodel_concat").output)
    
    neighbor_array = []
    for x in np.arange(5):
        prediction = extractor.predict([HSI_image] + metadata_data)
        neighbor_array.append(prediction)
    
    #stack and batch
    neighbor_array = np.vstack(neighbor_array)
    neighbor_array = np.expand_dims(neighbor_array, axis=0)
    
    neighbor_model = neighbors_model.create(ensemble_model = ensemble, freeze=False, k_neighbors=5, classes=2)
    prediction = neighbor_model.predict([HSI_image] + metadata_data + [neighbor_array])
    assert prediction.shape == (1,2)    