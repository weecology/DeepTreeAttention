from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def metadata_model(classes):
    # create model
    metadata_inputs = layers.Input(shape=(1,))
    x = Dense(classes, input_dim=1, activation='relu')(metadata_inputs)
    x = Dense(classes, input_dim=1, activation='relu')(x)
    output = Dense(classes, activation="softmax")(x)
    
    # Compile model
    return metadata_inputs, output