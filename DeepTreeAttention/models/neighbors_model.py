#Define spatial neighbor learning
import tensorflow as tf
from tensorflow.keras import backend as K

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
            
    n_features = ensemble_model.get_layer("submodel_concat").output.shape[1]
    input_shape = (k_neighbors, n_features)
    neighbor_inputs = tf.keras.layers.Input(shape=input_shape, name="neighbor_input")
    
    #original featuers from target tree
    original_features = ensemble_model.get_layer("submodel_concat").output
    
    #append to original inputs, add a dim to make it a matrix
    original_features_matrix = tf.keras.backend.expand_dims(original_features, axis=1)
    fused_inputs = tf.keras.backend.concatenate([neighbor_inputs,original_features_matrix],axis=1)

    #mask out zero padding if less than k_neighbors
    masked_inputs = tf.keras.layers.Masking(mask_value=0)(fused_inputs)
    
    key_features = tf.keras.layers.Dense(n_features, activation="relu",name="neighbor_feature_dense")(masked_inputs)
    key_features = tf.keras.backend.l2_normalize(key_features, axis=-1)
        
    #strip off previous head layers, target features are the HSI + metadata from the target tree
    query_features = tf.keras.layers.Dense(n_features, activation="relu",name="target_feature_dense")(original_features)
    query_features = tf.keras.backend.l2_normalize(query_features,axis=-1)  
    
    #Multiply to neighbor features
    #This may not be not right multiplication
    joined_features = tf.keras.layers.Dot(name="target_neighbor_multiply",axes=(1,2))([query_features, key_features])
    
    #Scale before softmax temperature (fixed at sqrt(112) for the moment)
    joined_features = tf.keras.layers.Lambda(lambda x: x/(0.1 *10.58))(joined_features)
    joined_features = tf.keras.layers.Softmax(name="Attention_softmax")(joined_features)
    
    #Skip connection for value features
    value_features = tf.keras.layers.Dense(n_features, activation="relu",name="skip_neighbor_feature_dense")(masked_inputs)
    context_vector = tf.keras.layers.Dot(name="lookup_function",axes=(1,1))([joined_features,value_features])
    context_vector = tf.keras.layers.Dense(n_features, name="context_vector", activation="relu")(context_vector)
    #context_vector = tf.keras.backend.l2_normalize(context_vector,axis=-1)  
    
    #Add as residual to original matrix normalized
    context_residual = tf.keras.layers.Add(name="ensemble_add_bias")([context_vector,original_features])
    
    merged_layers = tf.keras.layers.Dropout(0.7)(context_residual)
    output = tf.keras.layers.Dense(classes,name="ensemble_learn",activation="softmax")(merged_layers)
    
    return ensemble_model.inputs, neighbor_inputs, output

def create(ensemble_model, k_neighbors, classes, freeze=False, learning_rate=0.001):
    """Create a neighbor model based on a ensemble model
    Args:
        ensemble_model: see Hang2020_geographic.ensemble_model
        k_neighbors: number of neighbors in the array to use an input shape
        freeze: whether to freeze ensemble model layers and just train new top layer softmax
    Returns:
        model: a tf keras model for inference
    """
    
    ensemble_model_inputs, neighbor_inputs, output = define(ensemble_model=ensemble_model, k_neighbors=k_neighbors, freeze=freeze, classes=classes)
            
    neighbor_model = tf.keras.Model([ensemble_model_inputs, neighbor_inputs], output)
    
    metric_list = [tf.keras.metrics.CategoricalAccuracy(name="acc")]    
    
    neighbor_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)    
    
    return neighbor_model

