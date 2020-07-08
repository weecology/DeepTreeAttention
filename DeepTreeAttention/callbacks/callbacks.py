#Callbacks
"""Create training callbacks"""
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def on_train_begin(self):
        print("Starting training")

    def on_train_end(self):
        print("Stop training")

    def on_epoch_begin(self, epoch):
        print("Start epoch {} of training".format(epoch))

    def on_epoch_end(self, epoch):
        print("End epoch {} of training".format(epoch))
        
def create():
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=0, min_lr=0.0001, verbose=1)
    test_callback = CustomCallback()
    return [reduce_lr, test_callback]