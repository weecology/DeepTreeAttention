import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def model(classes):
    # create model
    #site label
    site_input = Input(shape(classes,),name="site_input")
    site_layers = Dense(classes*2, activation='relu')(site_input)
    
    
    #elevation
    metadata_inputs = Input(shape=(1,), name="metadata_input")    
    metadata_layer = Dense(classes*2, activation='relu')(metadata_inputs)
    
    joined_layer = tf.keras.layers.Concatenate()([metadata_layer, site_layers])
    x = Dense(classes, activation='relu', name="last_relu")(joined_layer)
    output = Dense(classes, activation="softmax")(x)
    
    # Compile model
    return metadata_inputs, output


def create(classes, learning_rate):
    inputs, outputs = model(classes)
    keras_model = tf.keras.Model(inputs,outputs)
    
    metric_list = [tf.keras.metrics.CategoricalAccuracy(name="acc")]    
    keras_model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)    
    
    return keras_model
    

    