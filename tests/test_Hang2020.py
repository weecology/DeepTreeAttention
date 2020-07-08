#test create model
from DeepTreeAttention.models.Hang2020 import create_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.zeros((1, 11, 11, 48), dtype=tf.keras.backend.floatx())

    return image

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [2, 10, 20])
def test_model(image, classes):
    model = create_model(classes=classes)
    prediction = model.predict(image)
    prediction.shape == (1, classes)

def test_multiply_broadcast():
    model = create_model(classes=20)
    
