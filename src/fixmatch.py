# Fixmatch datasets for strong and weak augmentation.
import os
import random

from src.models.multi_stage import TreeDataset
from src.utils import load_image

def lookup_year_image_path(annotations, year_lookup, individual, index, config):
    year_annotations = year_lookup[individual]
    selected_indices = [x for x in year_annotations if not x == index]
    random.shuffle(selected_indices)
    selected_year_path = annotations.iloc[selected_indices[0]].image_path
    selected_year_path = os.path.join(config["crop_dir"],selected_year_path)                    
    
    return selected_year_path
    
class FixmatchDataset(TreeDataset):
    """A strong and weak augmentation dataset
    Args:
       csv_file: path to csv file with image_path and label
       df: a pandas dataframe, mutually exclusive with csv_file
       config: a DeepTreeAttention config dict
       strong_augmentation: 'cross-year', for each image, choose the same location in a new year
    """
    def __init__(self, csv_file=None, df=None, config=None, strong_augmentation="cross_year"):
        super(FixmatchDataset, self).__init__(csv_file=csv_file, df=df, config=config, train=False)
        if strong_augmentation == "cross_year":
            self.year_lookup = self.annotations.groupby("individual").apply(lambda x: x.index.to_list()).to_dict()

    def __getitem__(self, index):
        individual, inputs = super().__getitem__(index)
       
        if self.config["preload_images"]:            
            if strong_augmentation == "cross_year":  
                year_annotations = self.year_lookup[individual]
                selected_indices = [x for x in year_annotations if not x == index]
                random.shuffle(selected_indices)
                strong_augmentation = self.image_dict[selected_indices[0]]            
        else:
            if strong_augmentation:
                selected_year_path = lookup_year_image_path(self.annotations, self.year_lookup, individual, index, self.config)
                strong_augmentation = load_image(selected_year_path, image_size=self.config["image_size"])
            
        inputs["Strong"] = strong_augmentation
        inputs["Weak"] = inputs["HSI"]
    
        return individual, inputs
