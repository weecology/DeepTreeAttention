#create model following
"""
Hyperspectral Image Classification with Attention Aided CNNs Hang et al. 2020
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def conv_module(x, K, kX, kY, stride, chanDim, padding="same"):
    """
    """
    # define a CONV => BN => RELU pattern
    x = Conv2D(K, (kX, kY), strides=stride, padding=padding)(x)
    x = BatchNormalization(axis=chanDim)(x)
    x = Activation("relu")(x)
    # return the block
    return x


def concat_attention(conv_layers, attention_layers):
    """
    Element-wise multiplication and padding
    """

    return conv_layers


def conv_and_attention_submodel(x, submodel):
    """Perform a set of convultions and attention and concat
    Args:
        x: functional keras model layers
        submodel: either "spectral" or "spatial" to get the right attention layer
    Returns:
        x: functional keras model layers
    """

    #Layers are a convoultional layer, an attention pooling layer and a concat
    conv_layers = conv_module(x, K, kX, kY, stride, chanDim)
    if submodel == "spectral":
        attention_layers = spectral_attention(conv_layer)
    elif submodel == "spatial":
        attention_layers = spatial_attention(conv_layer)
    else:
        raise ValueError("submodel must be either 'spectral' or 'spatial'")

    #combine
    x = concat_attention(conv_layers, attention_layers)

    return x


def spatial_network(x):
    """
    Learn spatial features with convolutional and attention pooling layers
    """
    #First submodel is 32 filters
    x = conv_and_attention_submodel(x, submodel="spatial")

    #Second submodel is 64 filters
    x = conv_and_attention_submodel(x, submodel="spatial")

    #Third submodel is 128 filters
    x = conv_and_attention_submodel(x, submodel="spatial")

    #This is still unclear, is it flatten and dense or just dense?
    x = layers.Dense(classes, activation="relu")

    return x


def spectral_network(x, classes=2):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    """

    #First submodel is 32 filters
    x = conv_and_attention_submodel(x, submodel="spectral")

    #Second submodel is 64 filters
    x = conv_and_attention_submodel(x, submodel="spectral")

    #Third submodel is 128 filters
    x = conv_and_attention_submodel(x, submodel="spectral")

    #This is still unclear, is it flatten and dense or just dense?
    x = layers.Dense(classes, activation="relu")

    return x


def spectral_attention(x):
    """
    """
    return x


def spatial_attention(x):
    """
    """
    return x


def weighed_consensus(spatial_layers, spectral_layer):

    return x


def model(height=48, width=11, depth=11, classes=2):
    """
    """
    input_shape = (height, width, depth)
    inputs = layers.Input(shape=input_shape)

    #spatial subnetwork
    spatial_layers = spatial_network(inputs)

    #spectral network
    spectral_layer = spectral_network(inputs)

    outputs = weighted_consensus(spatial_layers, spectral_layer)
    model = Model(inputs=inputs, outputs=outputs, name="DeepTreeAttention")

    return model
