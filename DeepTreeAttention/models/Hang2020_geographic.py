#Create spectral - spatial fusion model
from .layers import *


def create_model(height=11, width=11, channels=48, classes=2, sites=23, weighted_sum=False):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape)
    
    metadata_inputs = layers.Input(shape=(sites,))

    #spatial subnetwork and weak attention classifications
    spatial_attention_outputs = spatial_network(sensor_inputs, classes=classes)

    #spectral network
    spectral_attention_outputs = spectral_network(sensor_inputs, classes=classes)

    #Learn weighted average of just the final conv
    sensor_softmax = submodule_consensus(spatial_attention_outputs[2],
                                          spectral_attention_outputs[2],
                                          weighted_sum=weighted_sum)
    
    combined_softmax = metadata_layer(metadata_inputs, sensor_softmax, classes) 

    return sensor_inputs, metadata_inputs, combined_softmax, spatial_attention_outputs, spectral_attention_outputs
