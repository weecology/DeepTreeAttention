#Test crop_patches
import pytest
import numpy as np
import tensorflow as tf
from DeepTreeAttention.generators import extract_patches

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.random.rand(10, 10, 48)

    return image    

def test_extract_patches(image):
    """Extract patches and restitch"""
    height = 2
    width = 2
    
    patches = extract_patches.extract_patches(image, width=width, height=height)
    patches = tf.reshape(patches,[-1,width,height,48])
    
    assert len(patches) == 25
    assert patches[0,].shape == (width, height, 48)
    
    #reconstruct
    rows = tf.split(patches,image.shape[0]//height,axis=0)
    rows = [tf.concat(tf.unstack(x),axis=1) for x in rows] 
    
    reconstructed = tf.concat(rows,axis=0)   
    assert np.array_equal(image, reconstructed.numpy())
    