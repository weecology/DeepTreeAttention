#Create spectral - spatial fusion model
from .layers import *

def create_model(height=11, width=11, channels=48, classes=2, weighted_sum=False):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape)
    
    metadata_inputs = layers.Input(shape=(1,))

    #spatial subnetwork and weak attention classifications
    spatial_attention_outputs, spatial_attention_pool = spatial_network(sensor_inputs, classes=classes)

    #spectral network
    spectral_attention_outputs, spectral_attention_pool = spectral_network(sensor_inputs, classes=classes)

    #Learn weighted average of just the final conv
    sensor_softmax = submodule_consensus(spatial_attention_outputs[2],
                                          spectral_attention_outputs[2],
                                          weighted_sum=weighted_sum)
    
    metadata_activation = metadata_layer(metadata_inputs, classes)

    #Normalize all inputs before merging ensemble 
    spatial_relu = layers.Dense(classes * 4)(spatial_attention_pool[2])
    spectral_relu = layers.Dense(classes * 4)(spectral_attention_pool[2])
    
    combined_softmax = merge_softmax([metadata_activation, spatial_relu, spectral_relu], classes) 

    return sensor_inputs, metadata_inputs, combined_softmax, spatial_attention_outputs, spectral_attention_outputs
