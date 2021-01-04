#Define spatial neighbor learning
import tensorflow as tf

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
    
    #mask out zero padding if less than k_neighbors
    masked_inputs = tf.keras.layers.Masking()(neighbor_inputs)
    
    flatten_neighbors = tf.keras.layers.Flatten(name="flatten_inputs")(masked_inputs)
    neighbor_features = tf.keras.layers.Dense(n_features, activation="relu",name="neighbor_feature_dense")(flatten_neighbors)
    neighbor_features = tf.keras.backend.l2_normalize(neighbor_features)
        
    #strip off previous head layers, target features are the HSI + metadata from the target tree
    target_features = ensemble_model.get_layer("submodel_concat").output
    target_features = tf.keras.layers.Dense(n_features, activation="relu",name="target_feature_dense")(target_features)
    target_features = tf.keras.backend.l2_normalize(target_features)  
    
    #Multiply to neighbor features
    joined_features = tf.keras.layers.Multiply(name="target_neighbor_multiply")([target_features, neighbor_features])
    joined_features = tf.keras.layers.Softmax()(joined_features)
    
    #Skip connection for neighbor features
    neighbor_features = tf.keras.layers.Dense(n_features, activation="relu",name="skip_neighbor_feature_dense")(flatten_neighbors)
    joined_features = tf.keras.layers.Multiply()([joined_features, neighbor_features])
    output = tf.keras.layers.Dense(classes, name="neighbor_softmax", activation="softmax")(joined_features)
            
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
