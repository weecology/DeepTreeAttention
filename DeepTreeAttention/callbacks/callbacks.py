#Callbacks
"""Create training callbacks"""
from tensorflow.keras.callbacks import ReduceLROnPlateau

def create():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    return [reduce_lr]