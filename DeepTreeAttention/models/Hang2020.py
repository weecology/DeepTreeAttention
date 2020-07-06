#Create spectral - spatial fusion model
from .layers import *

def create_model(height=11, width=11, channels=48, classes=2, weighted_sum=False):
    """
    """
    input_shape = (height, width, channels)
    inputs = layers.Input(shape=input_shape)

    #spatial subnetwork
    spatial_layers = spatial_network(inputs, classes=classes)

    #spectral network
    spectral_layers = spectral_network(inputs, classes=classes)

    #Learn weighted average
    outputs = submodule_consensus(spatial_layers,
                                  spectral_layers,
                                  weighted_sum=weighted_sum)

    #outputs = layers.Average()([spatial_layers, spectral_layers])

    model = Model(inputs=inputs, outputs=outputs, name="DeepTreeAttention")

    return model