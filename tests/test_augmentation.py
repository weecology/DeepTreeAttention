#Test augmentation
import numpy as np
from src import augmentation
import torch
import pytest

@pytest.mark.parametrize("train",[True,False])
def test_augment(train):
    image = torch.randn(20, 369, 11, 11)    
    transformer = augmentation.augment(image_size=11, train=train)
    transformed_image = transformer(image)
    assert transformed_image.shape == image.shape
    if train:
        assert not np.array_equal(image, transformed_image)

def test_simulate_bands():
    image = torch.randn(369, 11, 11)    
    simulated_image = augmentation.simulate_bands(image, [10, 40, 100, 300])
    assert simulated_image.shape[0] == 3
    image[10:40,:,:].mean(dim=0) == simulated_image[0,:,:]