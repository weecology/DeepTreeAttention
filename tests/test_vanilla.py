#test vanilla model
#test create model
from DeepTreeAttention.models import vanilla
import pytest
import numpy as np
import tensorflow as tf

@pytest.fixture()
def HSI_image():
    image = np.zeros((1, 20, 20, 369), dtype=tf.keras.backend.floatx())
    return image

def test_create(HSI_image):
    vanilla_model = vanilla.create(height=20, width=20, channels=369, classes=2)
    prediction = vanilla_model.predict(HSI_image)