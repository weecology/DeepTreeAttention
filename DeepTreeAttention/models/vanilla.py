#Simplest baseline model for comparison
from tensorflow.keras import Model
from tensorflow.keras import layers

def create(height=11, width=11, channels=48, classes=2):
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape, name="data_input")
    x = layers.Conv2D(16, (3, 3), activation='relu')(sensor_inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.Flatten()(x)
    softmax = layers.Dense(classes, activation="softmax")(x)
    model = Model(sensor_inputs, softmax)

    return model
