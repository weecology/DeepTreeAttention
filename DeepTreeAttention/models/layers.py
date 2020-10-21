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


def spatial_network(x, classes=2):
    """
    Learn spatial features with alternating convolutional and attention pooling layers
    Args:
        x: keras.model object
        classes: number of label classes
    Returns:
        x: keras model object
        attention_outputs: list of the 3 attention softmax classification probabilities
    """
    #First submodel is 32 filters
    x = conv_module(x, K=32, maxpool=False)

    #Weak attention
    x, attention_1, pool_1 = spatial_attention(filters=32, classes=classes, x=x)

    #Second submodel is 64 filters - 2nd gets maxpool
    x = conv_module(x, K=64, maxpool=True)

    #Weak Attention
    x, attention_2, pool_2 = spatial_attention(filters=64, classes=classes, x=x)

    #Third submodel is 128 filters - 3rd gets maxpool
    x = conv_module(x, K=128, maxpool=True)

    #Weak Attention
    x, attention_3, pool_3 = spatial_attention(filters=128, classes=classes, x=x)

    return [attention_1, attention_2, attention_3], [pool_1,pool_2,pool_3]


def spectral_network(x, classes=2):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    """
    #First submodel is 32 filter
    x = conv_module(x, K=32, maxpool=False)
    x, attention_1, pool_1 = spectral_attention(filters=32, classes=classes, x=x)

    #Second submodel is 64 filters
    x = conv_module(x, K=64, maxpool=True)
    x, attention_2, pool_2 = spectral_attention(filters=64, classes=classes, x=x)

    #Third submodel is 128 filters
    x = conv_module(x, K=128, maxpool=True)
    x, attention_3, pool_3 = spectral_attention(filters=128, classes=classes, x=x)

    return [attention_1, attention_2, attention_3], [pool_1,pool_2,pool_3]


def spectral_attention(filters, classes, x):
    """
    Spectral attention layers: pool the feature maps and apply weak attention with a softmax multi-head output
    Args:
        filters: Number of incoming conv filters from main convolution blocks
        classes: Number of classes for one-hot softmax layers
        x: keras.model object
    Returns:
        x: keras.model object
        output: softmax attention layer
    """
    #Global average pool
    attention_layers = layers.GlobalAveragePooling2D()(x)
    attention_layers = layers.Reshape((1, 1, filters))(attention_layers)

    # Weak Attention with adaptive filter size based on depth of incoming feature map. Label 1,2,3 shallow -> deep
    if filters == 32:
        label = 1
        kernel_size = 3
    elif filters == 64:
        label = 2
        kernel_size = 5
    elif filters == 128:
        label = 3
        kernel_size = 7
    else:
        raise ValueError(
            "Unknown incoming kernel size {} for attention layers".format(kernel_size))

    attention_layers = layers.Conv2D(filters, (kernel_size, kernel_size),
                                     padding="SAME",
                                     activation="relu")(attention_layers)
    attention_layers = layers.Conv2D(filters, (kernel_size, kernel_size),
                                     padding="SAME",
                                     activation="sigmoid")(attention_layers)

    #Elementwise multiplication of attention with incoming feature map, expand among spatial dimension in 2D
    attention_layers = layers.Multiply()([x, attention_layers])

    #Add a classfication branch with max pool based on size of the layer
    if filters == 32:
        pool_size = (4, 4)
    elif filters == 64:
        pool_size = (2, 2)
    elif filters == 128:
        pool_size = (1, 1)
    else:
        raise ValueError("Unknown filter size for max pooling")

    class_pool = layers.MaxPool2D(pool_size)(attention_layers)
    class_pool = layers.Flatten(name="spectral_pooling_filters_{}".format(filters))(class_pool)
    output = layers.Dense(classes,
                          activation="softmax",
                          name="spectral_attention_{}".format(label))(class_pool)

    return attention_layers, output, class_pool


def spatial_attention(filters, classes, x):
    """
    Spatial attention layers channel pool the feature maps and apply weak attention
    Args:
        filters: Number of incoming conv filters from main convolution blocks
        classes: Number of classes for one-hot softmax layers
        x: keras.model object
    Returns:
        x: keras.model object
        output: softmax attention layer
    """
    #Channel pool
    attention_layers = layers.Conv2D(1, (1, 1), activation="relu")(x)

    # Weak Attention with adaptive kernel size based on size of incoming feature map
    if filters == 32:
        label = 1
        kernel_size = 7
    elif filters == 64:
        label = 2
        kernel_size = 5
    elif filters == 128:
        label = 3
        kernel_size = 3
    else:
        raise ValueError(
            "Unknown incoming kernel size {} for attention layers".format(kernel_size))

    attention_layers = layers.Conv2D(1, (kernel_size, kernel_size),
                                     padding="SAME",
                                     activation="relu")(attention_layers)
    attention_layers = layers.Conv2D(1, (kernel_size, kernel_size),
                                     padding="SAME",
                                     activation="sigmoid")(attention_layers)

    #Elementwise multiplication of attention with incoming feature map, expand among filter dimension in 3D
    attention_layers = layers.Multiply()([x, attention_layers])

    #Add a classfication branch with max pool based on size of the layer
    if filters == 32:
        pool_size = (4, 4)
    elif filters == 64:
        pool_size = (2, 2)
    elif filters == 128:
        pool_size = (1, 1)
    else:
        raise ValueError("Unknown filter size for max pooling")

    class_pool = layers.MaxPool2D(pool_size)(attention_layers)
    class_pool = layers.Flatten(name="spatial_pooling_filters_{}".format(filters))(class_pool)
    output = layers.Dense(classes,
                          activation="softmax",
                          name="spatial_attention_{}".format(label))(class_pool)

    return attention_layers, output, class_pool


class WeightedSum(layers.Layer):
    """A custom keras layer to learn a weighted sum of tensors"""

    def __init__(self, **kwargs):
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        self.a = self.add_weight(name='alpha',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(0.5),
                                 dtype='float32',
                                 trainable=True,
                                 constraint=tf.keras.constraints.min_max_norm(
                                     max_value=1, min_value=0))
        super(WeightedSum, self).build(input_shape)

    def call(self, model_outputs):
        return self.a * model_outputs[0] + (1 - self.a) * model_outputs[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


def submodule_consensus(spatial_layers, spectral_layers, weighted_sum=True):
    """Learned weighted sum among layers"""

    if weighted_sum:
        x = WeightedSum(name="weighted_sum")([spatial_layers, spectral_layers])
    else:
        x = layers.Add()([spatial_layers, spectral_layers])

    return x



class Weighted3Sum(layers.Layer):
    """A custom keras layer to learn a weighted sum of 3 tensors"""

    def __init__(self, **kwargs):
        super(Weighted3Sum, self).__init__(**kwargs)

    def build(self, input_shape=1):
        
        self.a = self.add_weight(name='alpha',
                                 shape=(1),
                                 initializer=tf.keras.initializers.Constant(0.5),
                                 dtype='float32',
                                 trainable=True)
        
        self.b = self.add_weight(name='beta',
                                     shape=(1),
                                     initializer=tf.keras.initializers.Constant(0.5),
                                     dtype='float32',
                                     trainable=True)
        
        self.g = self.add_weight(name='gamma',
                                     shape=(1),
                                         initializer=tf.keras.initializers.Constant(0.5),
                                         dtype='float32',
                                         trainable=True)
        
        super(Weighted3Sum, self).build(input_shape)

    def call(self, model_outputs):
        return (self.a * model_outputs[0]) +  (self.b * model_outputs[1]) + (self.g * model_outputs[2])

    def compute_output_shape(self, input_shape):
        return input_shape[0]