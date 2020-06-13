#Test crop_patches
import pytest
import numpy as np
import tensorflow as tf
from DeepTreeAttention.generators import extract_patches

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.zeros((10, 10, 48), dtype=tf.keras.backend.floatx())

    return image    

def test_extract_patches(image):
    """Extract patches and restitch"""
    patches = extract_patches.extract_patches(image, width=2, height=2)
    patches = tf.reshape(patches,[-1,2,2,48])
    
    assert len(patches) == 25
    assert patches[0,].shape == (2, 2, 48)
    