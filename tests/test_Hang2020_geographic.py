#test create model
from DeepTreeAttention.models import Hang2020_geographic as Hang
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def RGB_image():
    # create fake image input (only shape is used anyway) 
    image = np.zeros((1, 200, 200, 3), dtype=tf.keras.backend.floatx())
    metadata = np.ones((1,1))
    metadata = metadata.astype(float)
    return image, metadata

@pytest.fixture()
def HSI_image():
    # create fake image input (only shape is used anyway) 
    image = np.zeros((1, 20, 20, 369), dtype=tf.keras.backend.floatx())
    metadata = np.ones((1,1))
    metadata = metadata.astype(float)
    return image, metadata

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [3, 10, 20])
def test_model(RGB_image, classes):
    batch, height, width, channels = RGB_image[0].shape 
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=classes, height=height, width=width, channels=channels)    
    model = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    prediction = model.predict(RGB_image)
    assert prediction.shape == (1, classes)

def test_ensemble(RGB_image, HSI_image):    
    batch, height, width, channels = HSI_image[0].shape     
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=height, width=width, channels=channels)    
    model1 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    
    batch, height, width, channels = RGB_image[0].shape     
    sensor_inputs, sensor_outputs, spatial, spectral = Hang.define_model(classes=2, height=height, width=width, channels=channels)    
    model2 = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)    
    
    ensemble = Hang.ensemble(models=[model1, model2], classes=2)
    prediction = ensemble.predict([HSI_image[0], RGB_image[0]])
    assert prediction.shape == (1, 2)
    