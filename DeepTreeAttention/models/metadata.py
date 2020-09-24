from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

def metadata_model(classes):
    # create model
    model = Sequential()
    model.add(Dense(classes, input_dim=1, activation='relu'))
    model.add(Dense(classes, input_dim=1, activation='relu'))    
    model.add(Dense(classes, activation="softmax"))
    
    # Compile model
    model.compile(loss='acc', optimizer='adam')
    return model