# Fixmatch datasets for strong and weak augmentation.

# Dataset class
import os
import pandas as pd
from torch.utils.data import Dataset
from src import augmentation
from src.utils import *
import random

class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
    """
    def __init__(self, csv_file=None, df=None, config=None):
        if df is None:
            self.annotations = pd.read_csv(csv_file).reset_index(drop=True)
        else:
            self.annotations = df.reset_index(drop=True)
        self.config = config         
        self.image_size = config["image_size"]

        # Create augmentor
        self.weak_transformer = augmentation.train_augmentation()

        # Pin data to memory if desired
        if self.config["preload_images"]:
            self.image_dict = {}
            for index, row in self.annotations.iterrows():
                image_path = os.path.join(self.config["crop_dir"],row["image_path"])
                self.image_dict[index] = load_image(image_path, image_size=self.image_size)
        
        self.year_lookup = self.annotations.groupby("individual").apply(lambda x: x.index.to_list()).to_dict()
        
    def __len__(self):
        # 0th based index
        return self.annotations.index.max()

    def __getitem__(self, index):
        inputs = {}
        image_path = self.annotations.image_path.iloc[index]    
        individual = self.annotations.individual.iloc[index] 
        
        if self.config["preload_images"]:
            image = self.image_dict[index]
            weak_augmentation = self.weak_transformer(image)
            year_annotations = self.year_lookup[individual]
            selected_indices = [x for x in year_annotations if not x == index]
            random.shuffle(selected_indices)
            selected_year_image = self.image_dict[selected_indices[0]]
        else:
            image_basename = self.annotations.image_path.iloc[index]  
            image_path = os.path.join(self.config["crop_dir"],image_basename)                
            image = load_image(image_path, image_size=self.image_size)
        
            # Strong Augmentation is the same location in a different year
            year_annotations = self.year_lookup[individual]
            selected_indices = [x for x in year_annotations if not x == index]
            random.shuffle(selected_indices)
            selected_year_path = self.annotations.iloc[selected_indices[0]].image_path
            selected_year_path = os.path.join(self.config["crop_dir"],selected_year_path)                
      
            weak_augmentation = self.weak_transformer(image)    
            selected_year_image = load_image(selected_year_path, image_size=self.image_size)
        
        inputs["Strong"] = selected_year_image
        inputs["Weak"] = weak_augmentation
    
        return individual, inputs
