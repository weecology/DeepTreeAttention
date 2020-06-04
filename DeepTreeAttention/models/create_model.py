#create model following
"""
Hyperspectral Image Classification with Attention Aided CNNs Hang et al. 2020
"""
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers


def conv_module(x, K, kX=3, kY=3, chanDim=-1, padding="same", maxpool=False):
    """
    Basic convolutional block with batch norm and optional max pooling
    """
    # define a CONV => BN => RELU pattern
    x = layers.Conv2D(K, (kX, kY), padding=padding)(x)
    x = layers.BatchNormalization(axis=chanDim)(x)
    x = layers.Activation("relu")(x)
    if maxpool:
        x = layers.MaxPool2D()(x)
        
    return x


def concat_attention(conv_layers, attention_layers):
    """
    Element-wise multiplication and padding
    """

    return concat_layers


def conv_and_attention_submodel(x, K, submodel, attention=True, maxpool=False):
    """Perform a set of convultions and attention and concat
    Args:
        x: functional keras model layers
        K: convolutional filters
        submodel: either "spectral" or "spatial" to get the right attention layer
        maxpool: whether convolutional layers have a maxpool layer
    Returns:
        x: functional keras model layers
    """

    #Layers are a convoultional layer, an attention pooling layer and a concat
    x = conv_module(x, K, maxpool=maxpool)
    #add attention
    if attention:
        if submodel == "spectral":
            attention_layers = spectral_attention(x)
        elif submodel == "spatial":
            attention_layers = spatial_attention(x)
        else:
            raise ValueError("submodel must be either 'spectral' or 'spatial'")

        #combine
        x = concat_attention(x, attention_layers)

    return x


def spatial_network(x, attention=False, classes=2):
    """
    Learn spatial features with convolutional and attention pooling layers
    """
    #First submodel is 32 filters
    x = conv_and_attention_submodel(x, K=32, submodel="spatial", attention=attention)

    #Second submodel is 64 filters - 2nd gets maxpool
    x = conv_and_attention_submodel(x, K=64, submodel="spatial", attention=attention, maxpool=True)

    #Third submodel is 128 filters - 3rd gets maxpool
    x = conv_and_attention_submodel(x, K=128, submodel="spatial", attention=attention, maxpool=True)

    #This is still unclear, is it flatten and dense or just dense?
    x = layers.Dense(classes, activation="softmax")(x)

    return x


def spectral_network(x, attention=False, classes=2):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    """

    #First submodel is 32 filter
    x = conv_and_attention_submodel(x, K=32, submodel="spectral", attention=attention)

    #Second submodel is 64 filters
    x = conv_and_attention_submodel(x, K=64, submodel="spectral", attention=attention, maxpool=True)

    #Third submodel is 128 filters
    x = conv_and_attention_submodel(x, K=128, submodel="spectral", attention=attention, maxpool=True)

    #This is still unclear, is it flatten and dense or just dense?
    x = layers.Dense(classes, activation="softmax")(x)

    return x


def spectral_attention(x):
    """
    """
    return x


def spatial_attention(x):
    """
    """
    return x


def _weighted_sum(x):
    return tf.keras.backend.sum(x[0] * tf.keras.backend.expand_dims(x[1], -1), axis=1, keepdims=True)

class WeightedSum(layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""
    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(
            name='a',
            shape=(),
            initializer='ones',
            dtype='float32',
            trainable=True,
        )
        super(WeightedSum, self).build(input_shape)
        
    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]
    
def submodule_consensus(spatial_layers, spectral_layers, weighted=True):
    """Learned weighted sum among layers"""
    
    if weighted:
        x = WeightedSum()([spatial_layers, spectral_layers])
    else:
        x = layers.Average()([spatial_layers, spectral_layers])
    
    return x

def model(height=11, width=11, depth=48, classes=2):
    """
    """
    input_shape = (height, width, depth)
    inputs = layers.Input(shape=input_shape)

    #spatial subnetwork
    spatial_layers = spatial_network(inputs)

    #spectral network
    spectral_layers = spectral_network(inputs)
    
    #Still need to learn weights
    outputs = submodule_consensus(spatial_layers, spectral_layers, weighted=True)
    
    model = Model(inputs=inputs, outputs=outputs, name="DeepTreeAttention")

    return model
