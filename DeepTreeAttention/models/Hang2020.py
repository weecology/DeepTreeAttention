#Create spectral - spatial fusion model
from .layers import *

def create_model(height=11, width=11, channels=48, classes=2, weighted_sum=False):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    inputs = layers.Input(shape=input_shape)

    #spatial subnetwork and weak attention classifications
    spatial_layers, spatial_attention_outputs = spatial_network(inputs, classes=classes)

    #spectral network
    spectral_layers, spectral_attention_outputs = spectral_network(inputs,
                                                                   classes=classes)

    #Learn weighted average
    combined_output = submodule_consensus(spatial_layers,
                                          spectral_layers,
                                          weighted_sum=weighted_sum)

    #Output layers
    outputs = combined_output

    return inputs, combined_output, spatial_attention_outputs, spectral_attention_outputs
