#Create spectral - spatial fusion model
from .layers import *

def create_model(height=11, width=11, channels=48, classes=2):
    """
    """
    input_shape = (height, width, channels)
    inputs = layers.Input(shape=input_shape)

    #spatial subnetwork
    x = conv_module(inputs, K=32)
    x = layers.Flatten()(x)
    outputs = layers.Dense(classes, activation="softmax", name="softmax")(x)
    
    model = Model(inputs=inputs, outputs=outputs, name="DeepTreeAttention")

    return model