# Fixmatch datasets for strong and weak augmentation.

## The PCA Transformation was adopted from 
# Dataset class
import pandas as pd
import os
from torch.utils.data import Dataset
from src import augmentation
from src.utils import *

class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
    """
    def __init__(self, csv_file=None, df=None, config=None, train=True):
        if df is None:
            self.annotations = pd.read_csv(csv_file).reset_index(drop=True)
        else:
            self.annotations = df.reset_index(drop=True)
        self.train = train
        self.config = config         
        self.image_size = config["image_size"]

        # Create augmentor
        self.weak_transformer = augmentation.train_augmentation()
        self.strong_transformer = augmentation.train_augmentation()

        # Pin data to memory if desired
        if self.config["preload_images"]:
            self.image_dict = {}
            for index, row in self.annotations.iterrows():
                image_path = os.path.join(self.config["crop_dir"],row["image_path"])
                self.image_dict[index] = load_image(image_path, image_size=self.image_size)

            #pca_tensor = [value for key,value in self.image_dict.items()]
            #pca_tensor = torch.stack(pca_tensor)
            #self.strong_transformer.fit(pca_tensor)
        else:
            raise ValueError("Cannot use PCA tensor for augmentation and config['preload_images'] = False")
        
    def __len__(self):
        # 0th based index
        return self.annotations.shape[0]

    def __getitem__(self, index):
        inputs = {}
        image_path = self.annotations.image_path.loc[index]    
        individual = self.annotations.individual.loc[index] 
        
        if self.config["preload_images"]:
            image = self.image_dict[index]
        else:
            image_basename = self.annotations.image_path.loc[index]  
            image_path = os.path.join(self.config["crop_dir"],image_basename)                
            image = load_image(image_path, image_size=self.image_size)

        if self.train:
            strong_augmentation = self.strong_transformer(image)
            weak_augmentation = self.weak_transformer(image)
            
            inputs["Strong"] = strong_augmentation
            inputs["Weak"] = weak_augmentation
            
            label = self.annotations.label.loc[index]
            label = torch.tensor(label, dtype=torch.long)
            
            return individual, inputs, label
        else:
            inputs["HSI"] = image
            
            return individual, inputs