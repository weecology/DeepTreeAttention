#test create model
from DeepTreeAttention.models.metadata import metadata_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def sites():
    metadata = np.zeros((1,23))
    metadata[0,10] = 1
    metadata
    return metadata

#Test full model makes the correct number of predictions.
def test_model(sites):
    classes=74
    inputs, outputs= metadata_model(classes=classes, sites=len(sites[0,]))    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    prediction = model.predict(sites)
    
    assert prediction.shape == (1, classes)
