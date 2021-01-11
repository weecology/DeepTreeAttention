#test create model
from DeepTreeAttention.models import metadata
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def classes():
    return 74

@pytest.fixture()
def data():
    #simulate data
    elevation = np.random.random(1)
    sites = np.zeros(23)
    sites[8] = 1
    sites = np.expand_dims(sites,0)
    
    domain = np.zeros(15)
    domain[8] = 1
    domain = np.expand_dims(domain,0)
    
    return elevation, sites, domain

#Test full model makes the correct number of predictions.
def test_model(data, classes):
    model = metadata.create(classes=classes, sites=23, domains=15, learning_rate=0.001)
    prediction = model.predict(data)    
    assert prediction.shape == (1, classes)
