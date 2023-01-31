# Fixmatch datasets for strong and weak augmentation.
import random
from src.models.multi_stage import TreeDataset
import torch
class FixmatchDataset(TreeDataset):
    """A strong and weak augmentation dataset
    Args:
       csv_file: path to csv file with image_path and label
       df: a pandas dataframe, mutually exclusive with csv_file
       config: a DeepTreeAttention config dict
       strong_augmentation: 'cross-year', for each image, choose the same location in a new year, "spectral mask" masks n bands 
    """
    def __init__(self, csv_file=None, df=None, config=None, strong_augmentation="cross_year"):
        super(FixmatchDataset, self).__init__(csv_file=csv_file, df=df, config=config, train=False)
        self.strong_augmentation = strong_augmentation

    def __getitem__(self, index):
        individual, inputs = super().__getitem__(index)
        strong_augmentations = []
        if self.strong_augmentation == "debug":
            #Randomly select among years
            years = inputs["HSI"]
            strong_augmentations = years
            weak_augmentations = years
        
        if self.strong_augmentation == "cross_year":
            #Randomly select among years
            years = inputs["HSI"]
            swapped_order = years.copy()
            random.shuffle(swapped_order)
            strong_augmentations = swapped_order
            weak_augmentations = years

        if self.strong_augmentation == "random_mask":
            #Randomly select among years
            years = inputs["HSI"]
            strong_augmentations = []
            drop = torch.nn.Dropout2d()
            for image in years:
                dropout_image = drop(image)
                strong_augmentations.append(dropout_image)                
            weak_augmentations = years
            
        inputs["Strong"] = strong_augmentations
        inputs["Weak"] = weak_augmentations
        

        return individual, inputs
