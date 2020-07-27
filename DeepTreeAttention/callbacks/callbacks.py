#Callbacks
"""Create training callbacks"""
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback
import os
from datetime import datetime

def tensorboard_callback(log_dir):
    # Create a TensorBoard callback
    tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = log_dir,
                                                     histogram_freq = 1,
                                                     profile_batch = '500,520')
    
def create(log_dir=None):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.1,
                                  patience=5,
                                  min_lr=0.00001,
                                  verbose=1)
    
    if log_dir:
        tensorboard  = tensorboard_callback(log_dir)
    
    return [reduce_lr, tensorboard]
