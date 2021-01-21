import tensorflow as tf
from tensorflow.keras.layers import Dense, Input

def model(classes, sites, domains):
    # create model
    #site label
    site_input = Input(shape=(sites,),name="site_input")
    site_layers = tf.keras.layers.BatchNormalization()(site_input)
    
    domain_input = Input(shape=(domains,),name="domain_input")
    domain_layers = tf.keras.layers.BatchNormalization()(domain_input)
    
    #elevation
    elevation_input = Input(shape=(1,), name="elevation_input")    
    elevation_layer = tf.keras.layers.BatchNormalization()(elevation_input)
    
    #Bottleneck layer size should be the same as the concat features
    joined_layer = tf.keras.layers.Concatenate()([elevation_layer, site_layers, domain_layers])
    
    x = Dense(classes, activation='relu', name="last_relu")(joined_layer)
    output = Dense(classes, activation="softmax")(x)
    
    return elevation_input, site_input, domain_input, output


def create(classes, sites, domains, learning_rate):
    elevation_input, site_input, domain_input, output= model(classes, sites, domains)
    keras_model = tf.keras.Model([elevation_input, site_input, domain_input],output)
    
    metric_list = [tf.keras.metrics.CategoricalAccuracy(name="acc")]    
    keras_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)    
    
    return keras_model
    

    