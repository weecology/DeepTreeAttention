#test create model
from DeepTreeAttention.models import layers
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
    output = layers.conv_module(image, K=32, maxpool=maxpool)
    assert output.shape == expected