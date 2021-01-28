#Define spatial neighbor learning
import tensorflow as tf
import math
from DeepTreeAttention.models.layers import WeightedSum

def define(ensemble_model, k_neighbors, classes=2, freeze=False):
    """Define a neighbor model based on a ensemble model
    Args:
        ensemble_model: see Hang2020_geographic.ensemble_model
        k_neighbors: number of neighbors in the array to use an input shape
        freeze: whether to freeze ensemble model layers and just train new top layer softmax
    Returns:
        model: a tf keras model for inference
    """
    #Neighbor input    
    #shape is equal to the concat shape of the ensemble model
    if freeze:
        for x in ensemble_model.layers:
            x.trainable=False
            
    input_shape = (k_neighbors, classes)
    neighbor_inputs = tf.keras.layers.Input(shape=input_shape, name="neighbor_input")
    
    neighbor_distances = tf.keras.layers.Input(shape=(k_neighbors), name="neighbor_distance_input")
    context_vector = tf.keras.layers.MaxPooling1D(pool_size=k_neighbors)(neighbor_inputs)
    context_vector = tf.keras.layers.Dense(classes)(context_vector)
    
    #original featuers from target tree
    original_features = ensemble_model.get_layer("ensemble_learn").output

    ##scale by confidence of initial prediction. 
    #previous_confidence = tf.math.reduce_max(original_features)
    #scaled_context = tf.divide(context_vector, previous_confidence)
    
    ##Squueze 1st dim for addition with original features
    scaled_context = tf.keras.backend.squeeze(context_vector,1)
    
    #Add as residual to original matrix normalized
    context_residual = WeightedSum(name="ensemble_add_bias")([scaled_context,original_features])      
    context_residual = tf.keras.layers.Dense(classes)(context_residual)
    output = tf.keras.layers.Softmax(name="neighbor_softmax")(context_residual)

    return ensemble_model.inputs, neighbor_inputs, neighbor_distances, output

def create(ensemble_model, k_neighbors, classes, freeze=False, learning_rate=0.001):
    """Create a neighbor model based on a ensemble model
    Args:
        ensemble_model: see Hang2020_geographic.ensemble_model
        k_neighbors: number of neighbors in the array to use an input shape
        freeze: whether to freeze ensemble model layers and just train new top layer softmax
    Returns:
        model: a tf keras model for inference
    """
    
    ensemble_model_inputs, neighbor_inputs, distances, output = define(ensemble_model=ensemble_model, k_neighbors=k_neighbors, freeze=freeze, classes=classes)
            
    neighbor_model = tf.keras.Model([ensemble_model_inputs, neighbor_inputs, distances], output)
    
    metric_list = [tf.keras.metrics.CategoricalAccuracy(name="acc")]    
    
    neighbor_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)    
    
    return neighbor_model

