#test create model
from DeepTreeAttention.models.Hang2020_geographic import define_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) 
    image = np.zeros((1, 11, 11, 48), dtype=tf.keras.backend.floatx())
    metadata = np.ones((1,1))
    metadata = metadata.astype(float)
    return image, metadata

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [3, 10, 20])
def test_model(image, classes):
    sensor_inputs, sensor_outputs, spatial, spectral = define_model(classes=classes)    
    model = tf.keras.Model(inputs=sensor_inputs, outputs=sensor_outputs)
    prediction = model.predict(image)
    prediction.shape == (1, classes)
