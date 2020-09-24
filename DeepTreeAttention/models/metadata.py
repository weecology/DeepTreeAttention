from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def metadata_model(classes):
    # create model
    metadata_inputs = Input(shape=(1,))
    x = Dense(classes, input_dim=1, activation='relu')(metadata_inputs)
    x = Dense(classes, input_dim=1, activation='relu')(x)
    output = Dense(classes, activation="softmax")(x)
    
    # Compile model
    return metadata_inputs, output