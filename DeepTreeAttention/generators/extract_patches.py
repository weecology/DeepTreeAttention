##Generate patches from a large raster##
"""preprocessing model for creating a non-overlapping sliding window of fixed size to generate tfrecords for model training"""
import rasterio
import tensorflow as tf
import numpy as np

def _read_file(path):
    """Read a hyperspetral raster .tif 
    Args:
        path: a path to a .tif hyperspectral raster
    Returns:
        src: a numpy array of height x width x channels
        """
    r = rasterio.open()
    src = r.read()
    
    return src

def extract_patches(image, width, height):
    # The size of sliding window
    ksizes = [1, width, height, 1] 
    
    # How far the centers of 2 consecutive patches are in the image
    strides = [1, width, height, 1]
    
    # The document is unclear. However, an intuitive example posted on StackOverflow illustrate its behaviour clearly. 
    # http://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
    rates = [1, 1, 1, 1] # sample pixel consecutively
    
    # padding algorithm to used
    padding='VALID' # or 'SAME'
    
    image = tf.expand_dims(image, 0)
    image_patches = tf.image.extract_patches(image, ksizes, strides, rates, padding)   
    
    #Squeeze batch array
    image_patches = tf.squeeze(image_patches)       
    
    return image_patches