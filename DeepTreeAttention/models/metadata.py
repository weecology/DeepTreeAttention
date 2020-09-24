from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input

def metadata_model(classes,sites):
    # create model
    metadata_inputs = Input(shape=(sites,))
    x = Dense(classes, activation='relu')(metadata_inputs)
    x = Dense(classes, activation='relu')(x)
    output = Dense(classes, activation="softmax")(x)
    
    # Compile model
    return metadata_inputs, output