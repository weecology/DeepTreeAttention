#Create spectral - spatial fusion model
from .layers import *
from tensorflow.keras import metrics

def define_model(height=11, width=11, channels=48, classes=2, weighted_sum=False, softmax=True):
    """
    Create model and return output layers to allow training at different levels
    """
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape)
    
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
        weighted_sum=weighted_sum)

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

def ensemble(models, classes, freeze=True):
    
    #freeze orignal model layers
    if freeze:
        for model in models:
            for x in model.layers:
                x.trainable = False
    
    #Take joint inputs, ASSUMES that this is a single input model 
    inputs = [x.inputs[0] for x in models]
    
    #Reduce bottleneck layer
    decap_models = []
    for index, model in enumerate(models):
        new_model = tf.keras.Model(inputs=model.inputs, outputs = model.outputs)
        for x in new_model.layers:
            x._name = x.name + str(index)
        decap_models.append(new_model.output)
        
    #concat and learn ensemble weights
    merged_layers = WeightedSum(name="weighted_sum")(decap_models)
    ensemble_model = tf.keras.Model(inputs=inputs,
                           outputs=merged_layers,
                                name="ensemble_model")    
    
    return ensemble_model