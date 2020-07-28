#Callbacks
"""Create training callbacks"""
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, TensorBoard
import os
from datetime import datetime
    
def create(log_dir=None):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  cooldown=5,
                                  min_lr=0.00001,
                                  verbose=1)
    
    if log_dir is not None:
        tensorboard = TensorBoard(log_dir = log_dir,
                                                     histogram_freq = 1,
                                                     profile_batch = 10)
    
    return [reduce_lr]
