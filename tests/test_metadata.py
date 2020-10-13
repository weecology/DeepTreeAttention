#test create model
from DeepTreeAttention.models import metadata
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def classes():
    return 74

@pytest.fixture()
def data(classes):
    #simulate data
    metadata = np.random.random(1)
    return metadata

#Test full model makes the correct number of predictions.
def test_model(data, classes):
    model = metadata.create(classes,learning_rate=0.001)
    prediction = model.predict(data)    
    assert prediction.shape == (1, classes)
