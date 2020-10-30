#test create model
from DeepTreeAttention.models import Hang2020_geographic as Hang
from DeepTreeAttention.models import metadata
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def metadata_data():
    #simulate data
    elevation = np.random.random(1)
    height = np.random.random(1)
    
    sites = np.zeros(10)
    sites[8] = 1
    sites = np.expand_dims(sites,0)
    return [elevation, height, sites]

@pytest.fixture()
def RGB_image():
    image = np.zeros((1, 200, 200, 3), dtype=tf.keras.backend.floatx())
    return image

@pytest.fixture()
def HSI_image():
    image = np.zeros((1, 20, 20, 369), dtype=tf.keras.backend.floatx())
    return image

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [3, 10, 20])
def test_model(RGB_image, classes):
    batch, height, width, channels = RGB_image.shape 
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=classes, height=height, width=width, channels=channels)    
    model = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    prediction = model.predict(RGB_image)
    assert prediction.shape == (1, classes)

def test_ensemble(RGB_image, HSI_image, metadata_data):    
    batch, height, width, channels = HSI_image.shape     
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=height, width=width, channels=channels)    
    model1 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    
    batch, height, width, channels = RGB_image.shape     
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=height, width=width, channels=channels)    
    model2 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)    
    
    metadata_model = metadata.create(classes=2, sites=10, learning_rate=0.001)
    ensemble = Hang.learned_ensemble(HSI_model=model1, RGB_model=model2, metadata_model=metadata_model, classes=2)
    prediction = ensemble.predict([HSI_image, RGB_image] + metadata_data)
    assert prediction.shape == (1, 2)
        