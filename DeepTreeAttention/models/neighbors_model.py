#Define spatial neighbor learning
import tensorflow as tf
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
    
    #original featuers from target tree
    original_features = ensemble_model.get_layer("ensemble_learn").output
                
    #mask out zero padding if less than k_neighbors
    masked_inputs = tf.keras.layers.Masking(mask_value=0)(neighbor_inputs)
    
    key_features = tf.keras.layers.Dense(classes, activation="relu",name="neighbor_feature_dense")(masked_inputs)
        
    #strip off previous head layers, target features are the HSI + metadata from the target tree
    query_features = tf.keras.layers.Dense(classes, activation="relu",name="target_feature_dense")(original_features)
    
    #Multiply to neighbor features
    #This may not be not right multiplication
    joined_features = tf.keras.layers.Dot(name="target_neighbor_multiply",axes=(1,2))([query_features, key_features])
    
    #Scale before softmax temperature (fixed at sqrt(112) for the moment)
    joined_features = tf.keras.layers.Lambda(lambda x: x/(0.1 *10.58))(joined_features)
        
    #Zero out any masked entries
    attention_weights = tf.keras.layers.Softmax(name="Attention_softmax")(joined_features)
    
    #Skip connection for value features
    value_features = tf.keras.layers.Dense(classes, activation="relu",name="skip_neighbor_feature_dense")(masked_inputs)
    
    context_vector = tf.keras.layers.Dot(name="lookup_function",axes=(1,1))([attention_weights,value_features])
    context_vector = tf.keras.layers.Dense(classes, name="context_vector", activation="relu")(context_vector)
    
    #Add as residual to original matrix normalized
    context_residual = WeightedSum(name="ensemble_add_bias")([context_vector,original_features])
    
    dropout_layers = tf.keras.layers.Dropout(0.8,name="context_dropout")(context_residual)
    output = tf.keras.layers.Dense(classes,name="neighbor_softmax",activation="softmax")(dropout_layers)
    
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

