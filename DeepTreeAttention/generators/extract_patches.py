##Generate patches from a large raster##
"""preprocessing model for creating a non-overlapping sliding window of fixed size to generate tfrecords for model training"""
import rasterio
import tensorflow as tf
import numpy as np


def extract_patches(image, width, height):
    # The size of sliding window
    ksizes = [1, width, height, 1]

    # Move over 1 pixel and make a new patch
    strides = [1, 1, 1, 1]

    # The document is unclear. However, an intuitive example posted on StackOverflow illustrate its behaviour clearly.
    # http://stackoverflow.com/questions/40731433/understanding-tf-extract-image-patches-for-extracting-patches-from-an-image
    rates = [1, 1, 1, 1]  # sample pixel consecutively

    # padding algorithm to used
    padding = 'SAME'  # or 'SAME'

    image = tf.expand_dims(image, 0)
    image_patches = tf.image.extract_patches(image, ksizes, strides, rates, padding)

    #Squeeze batch array
    image_patches = tf.squeeze(image_patches)

    return image_patches
