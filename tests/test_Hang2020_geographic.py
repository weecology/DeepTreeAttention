#test create model
from DeepTreeAttention.models.Hang2020_geographic import create_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.zeros((1, 11, 11, 48), dtype=tf.keras.backend.floatx())
    metadata = np.ones((1,1))
    metadata = metadata.astype(float)
    return image, metadata

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [2, 10, 20])
def test_model(image, classes):
    sensor_inputs, metadata_inputs, combined_output, spatial, spectral = create_model(classes=classes)    
    model = tf.keras.Model(inputs=[sensor_inputs, metadata_inputs], outputs=combined_output)
    prediction = model.predict(image)
    prediction.shape == (1, classes)
