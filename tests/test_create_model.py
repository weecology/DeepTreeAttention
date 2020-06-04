#test create model
from DeepTreeAttention.models import create_model
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.zeros((1, 11, 11, 48), dtype=tf.keras.backend.floatx())

    return image


#Test conv layers pooling
@pytest.mark.parametrize("maxpool,expected", [(True, (1, 5, 5, 32)),
                                              (False, (1, 11, 11, 32))])
def test_conv_module(image, maxpool, expected):
    output = create_model.conv_module(image, K=32, maxpool=maxpool)
    assert output.shape == expected


#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [2, 10, 20])
def test_create_model(image, classes):
    model = create_model.model(classes=classes)
    prediction = model.predict(image)
    prediction.shape == (1, classes)
