#Create spectral - spatial fusion model
from .layers import *


def create_model(height=11, width=11, channels=48, classes=2):
    """
    """
    input_shape = (height, width, channels)
    inputs = layers.Input(shape=input_shape)
    spatial_layers = spatial_network(inputs, classes=classes)
    model = Model(inputs=inputs, outputs=spatial_layers, name="DeepTreeAttention_spatial")

    return model
