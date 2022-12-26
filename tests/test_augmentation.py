#Test augmentation
import numpy as np
from src import augmentation
from src import visualize
import torch

def test_train_augmentation():
    image = torch.randn(20, 369, 11, 11)    
    transformer = augmentation.train_augmentation()
    transformed_image = transformer(image)
    assert transformed_image.shape == image.shape
    assert not np.array_equal(image, transformed_image)

def test_PCATransformation():
    images = torch.randn(20, 369, 11, 11)    
    transformer = augmentation.PCATransformation()
    transformer.fit(images)
    transformed_images = transformer.transform(data=images)
    
    assert transformed_images.shape == images.shape
    assert not np.array_equal(images, transformed_images)
    
