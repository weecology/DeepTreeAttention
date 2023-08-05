#Test augmentation
import numpy as np
from src import augmentation
import torch
import pytest

@pytest.mark.parametrize("train",[True,False])
def augment(train):
    image = torch.randn(20, 369, 11, 11)    
    transformer = augmentation.augment(image_size=11, train=train)
    transformed_image = transformer(image)
    assert transformed_image.shape == image.shape
    if train:
        assert not np.array_equal(image, transformed_image)