#test create model
from DeepTreeAttention.models.single_conv import create_model
import pytest
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

@pytest.fixture()
def image():
    # create fake image input (only shape is used anyway) # logic from https://github.com/fizyr/tf-retinanet/blob/master/tests/layers/test_misc.py
    image = np.zeros((1, 11, 11, 48), dtype=tf.keras.backend.floatx())

    return image

#Test full model makes the correct number of predictions.
@pytest.mark.parametrize("classes", [2, 10, 20])
def test_model(image, classes):
    model = create_model(classes=classes)
    prediction = model.predict(image)
    prediction.shape == (1, classes)

#Test that the loss drops
def test_loss():
    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label
    
    (ds_train, ds_test), ds_info = tfds.load(
        'mnist',
        split=['train', 'test'],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    
    ds_train = ds_train.map(
        normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    ds_train = ds_train.shuffle(100)
    ds_train = ds_train.batch(128)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)    
    
    #Create tiny subset
    ds_train = ds_train.take(10)    
    ds_test = ds_test.take(5)
    
    model = create_model(height=28, width =28, channels=1, classes=10)
    
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(0.001),
        metrics=['accuracy'],
    )
    
    history = model.fit(
        ds_train,
        epochs=2
    )    
    
    #assert that loss drops
    assert history.history["loss"][-1] < history.history["loss"][0]