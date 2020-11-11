#Create spectral - spatial fusion model
from .layers import *
from tensorflow.keras import metrics

def define_model(height=11, width=11, channels=48, classes=2, weighted_sum=False, softmax=True):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape, name="data_input")
    
    #spatial subnetwork and weak attention classifications
    spatial_attention_outputs, spatial_attention_pool = spatial_network(sensor_inputs, classes=classes)

    #spectral network
    spectral_attention_outputs, spectral_attention_pool = spectral_network(sensor_inputs, classes=classes)

    #Learn weighted average of just the final conv
    if softmax:
        sensor_output = submodule_consensus(
            spatial_attention_outputs[2],
            spectral_attention_outputs[2],
            weighted_sum=weighted_sum)
    else:
        sensor_output = submodule_consensus(
            spatial_attention_pool[2],
            spectral_attention_pool[2],
            weighted_sum=weighted_sum)
        
    return sensor_inputs, sensor_output, spatial_attention_outputs, spectral_attention_outputs

def create_models(height, width, channels, classes, learning_rate, weighted_sum=True):
    #Define model structure
    sensor_inputs, sensor_outputs, spatial_attention_outputs, spectral_attention_outputs = define_model(
        height = height,
        width = width,
        channels = channels,
        classes = classes,
        weighted_sum=weighted_sum, softmax=True)

    #Full model compile
    model = tf.keras.Model(inputs=sensor_inputs,
                                outputs=sensor_outputs,
                                name="DeepTreeAttention")

    #compile full model
    metric_list = [metrics.CategoricalAccuracy(name="acc")]    
    model.compile(loss="categorical_crossentropy",
                       optimizer=tf.keras.optimizers.Adam(
                           lr=float(learning_rate)),
                       metrics=metric_list)
    #compile
    loss_dict = {
        "spatial_attention_1": "categorical_crossentropy",
        "spatial_attention_2": "categorical_crossentropy",
        "spatial_attention_3": "categorical_crossentropy"
    }

    # Spatial Attention softmax model
    spatial_model = tf.keras.Model(inputs=sensor_inputs,
                                        outputs=spatial_attention_outputs,
                                        name="DeepTreeAttention")

    spatial_model.compile(
        loss=loss_dict,
        loss_weights=[0.01, 0.1, 1],
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)

    # Spectral Attention softmax model
    spectral_model = tf.keras.Model(inputs=sensor_inputs,
                                         outputs=spectral_attention_outputs,
                                         name="DeepTreeAttention")

    #compile loss dict
    loss_dict = {
        "spectral_attention_1": "categorical_crossentropy",
        "spectral_attention_2": "categorical_crossentropy",
        "spectral_attention_3": "categorical_crossentropy"
    }

    spectral_model.compile(
        loss=loss_dict,
        loss_weights=[0.01, 0.1, 1],
        optimizer=tf.keras.optimizers.Adam(
            lr=float(learning_rate)),
        metrics=metric_list)
    
    return model, spatial_model, spectral_model

def strip_sensor_softmax(model, classes, index, squeeze=False, squeeze_size=128):
    #prepare RGB model
    spectral_relu_layer = model.get_layer("spectral_pooling_filters_128").output
    spatial_relu_layer = model.get_layer("spatial_pooling_filters_128").output
    weighted_relu = WeightedSum(name="within_model_weighted_" + index)([spectral_relu_layer, spatial_relu_layer])
    
    if squeeze:
        weighted_relu = layers.Dense(squeeze_size)(weighted_relu)
        
    stripped_model = tf.keras.Model(inputs=model.inputs, outputs = weighted_relu)
    for x in model.layers:
        x._name = x.name + str(index)
    
    return stripped_model

def learned_ensemble(RGB_model, HSI_model, metadata_model, classes, freeze=True):
    
    #Strip to last pooling layer    
    stripped_HSI_model = strip_sensor_softmax(HSI_model, classes, index = "HSI", squeeze=True, squeeze_size=classes)    
    stripped_RGB_model = strip_sensor_softmax(RGB_model, classes, index = "HSI", squeeze=True, squeeze_size=classes)          
    normalized_metadata = layers.BatchNormalization()(metadata_model.get_layer("last_relu").output)
    stripped_metadata = tf.keras.Model(inputs=metadata_model.inputs, outputs = normalized_metadata)
    
    #make names unique
    for x in stripped_RGB_model.layers:
        x._name = x.name + str("RGB")
    
    for x in stripped_HSI_model.layers:
        x._name = x.name + str("HSI")    
    
    #concat and learn ensemble weights
    HSI_meta_fuse = WeightedSum(name="HSI_meta_weighted_sum")([stripped_HSI_model.output, stripped_metadata.output])    
    HSI_meta_fuse = layers.Dropout(0.7)(HSI_meta_fuse)
    
    HSI_RGB_fuse = WeightedSum(name="HSI_RGB_weighted_sum")([stripped_HSI_model.output, stripped_RGB_model.output])    
    HSI_RGB_fuse = layers.Dropout(0.7)(HSI_RGB_fuse)    
    
    merged_layers = layers.Concatenate()([HSI_meta_fuse, HSI_RGB_fuse, stripped_HSI_model.output])
    merged_layers = layers.Dense(classes*3,name="ensemble_learn1",activation="relu")(merged_layers)
    merged_layers = layers.Dense(classes*2,name="ensemble_learn2",activation="relu")(merged_layers)
    ensemble_softmax = layers.Dense(classes,name="ensemble_sotmax",activation="softmax")(merged_layers)

    #Take joint inputs    
    ensemble_model = tf.keras.Model(inputs=HSI_model.inputs+RGB_model.inputs+metadata_model.inputs,
                                    outputs=ensemble_softmax,
                           name="ensemble_model")    
    
    return ensemble_model

def weighted_ensemble(RGB_model, HSI_model, metadata_model, classes, freeze=True):
    
    #freeze orignal model layers
    if freeze:
        for model in [RGB_model, HSI_model, metadata_model]:
            for x in model.layers:
                x.trainable = False
    
    for x in RGB_model.layers:
        x._name = x.name + str("RGB")
    
    for x in HSI_model.layers:
        x._name = x.name + str("HSI")
           
    merged_layers = layers.Concatenate()(name="ensemble_weight")([HSI_model.output, RGB_model.output, metadata_model.output])
    ensemble_softmax = layers.Dense(classes, name="ensemble_softmax")(merged_layers)
    
    #Take joint inputs    
    ensemble_model = tf.keras.Model(inputs=HSI_model.inputs+RGB_model.inputs+metadata_model.inputs,
                           outputs=ensemble_softmax,
                                name="ensemble_model")    
    
    return ensemble_model