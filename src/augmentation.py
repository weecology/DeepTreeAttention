#Training HSI Augmentations 
from torchvision import transforms
import numpy as np
from typing import List
from sklearn.decomposition import PCA

def train_augmentation():
    """Torchvision transforms
    Args:
        image: a torch vision tensor with C, H, W order
    Returns:
        transformed_image: an augmentated image
    """
    
    transform_list = []
    transform_list.append(transforms.RandomHorizontalFlip(p=1))
    transform_list.append(transforms.RandomVerticalFlip(p=1))
    
    return transforms.Compose(transform_list)

SAMPLES_COUNT = 0

class PCATransformation():
    """
    Transform samples using PCA, modify first component by multiplying
    it by a random value from a given range and then inverse
    transform principal components back to the original domain.
    Adapted from https://github.com/ESA-PhiLab/hypernet/blob/master/python_research/augmentation/transformations.py. 
    """
    def __init__(self, n_components: float=2, low=0.9, high=1.1):
        """
        :param n_components: Number of components to be returned by PCA
        transformation
        :param low: Lower boundary of the random value range
        :param high: Upper boundary of the random value range
        """
        self.pca = PCA(n_components=n_components)
        self.low = low
        self.high = high

    def transform(self, data: np.ndarray, transformations_count: int=4) \
            -> np.ndarray:
        """
        Transform samples
        :param data: Data to be transformed
        :param transformations_count: Number of transformations for each class
        :return:
        """
        reshaped_data = data.flatten().reshape(1,-1)        
        transformed = self.pca.transform(reshaped_data)
        for x in range(transformations_count):
            random_values = np.random.uniform(low=self.low, high=self.high,
                                          size=transformed.shape)
            transformed = transformed + random_values
            
        transformed = self.pca.inverse_transform(transformed).astype(np.float32)
        
        return transformed.reshape(data.shape)
        
    def fit(self, data: np.ndarray) -> None:
        """
        Fit PCA to data
        :param data: Data to fit to
        :return: None
        """
        reshaped_data = data.reshape(-1, data.shape[1]*data.shape[2] * data.shape[3])
        self.pca = self.pca.fit(reshaped_data)

