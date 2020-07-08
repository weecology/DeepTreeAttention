#Callbacks
"""Create training callbacks"""
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from keras.callbacks import Callback
from sklearn.metrics import f1_score

def create():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0, min_lr=0.0001, verbose=1)
    return [reduce_lr]