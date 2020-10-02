#test create model
from DeepTreeAttention.models.metadata import metadata_model
import pytest
import numpy as np
import tensorflow as tf

#@pytest.fixture()
#def sites():
    ##simulate data
    #metadata = [0.1, np.zeros((1,23))]
    #metadata[1][0,10] = 1
    #return metadata

##Test full model makes the correct number of predictions.
#def test_model(sites):
    #classes=74
    #inputs, outputs= metadata_model(classes=classes, sites=23)  
    #model = tf.keras.Model(inputs=inputs, outputs=outputs)
    #prediction = model.predict(sites)    
    #assert prediction.shape == (1, classes)
