#Year model
from torch import nn
import torch
from src.models import Hang2020
from src import augmentation
import os
from src import utils
import pandas as pd
from torch.utils.data import Dataset

class learned_ensemble(nn.Module):
    def __init__(self, years, classes, config):
        """A weighted ensemble of year models
        Args:
            years (list): a list of numeric years e.g [2019, 2020, 2021]
        """
        super().__init__()
        self.year_models = nn.ModuleDict()
        for year in years:
            #Load from state dict of previous run
            if config["pretrain_state_dict"]:
                base_model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
            else:
                base_model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=classes)
            
            self.year_models[str(year)] = base_model
        
    def forward(self, images):
        year_scores = []
        for year, image in images.items():
            # Skip padding or no_data tensors
            if image.sum() == 0:
                continue
            try:
                score = self.year_models[year](image)
            except KeyError:
                print("key {} is not in model dict {}".format(year, self.year_models.keys()))
                continue
            year_scores.append(score)

        if len(year_scores) == 0:
            return None

        return torch.stack(year_scores, axis=1).mean(axis=1)
                
    
class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
    """
    def __init__(self, df=None, csv_file=None, config=None, train=True, image_dict=None):
        if csv_file:
            self.annotations = pd.read_csv(csv_file)
        else:
            self.annotations = df
        
        self.train = train
        self.config = config         
        self.image_size = config["image_size"]
        self.years = self.annotations.tile_year.unique()
        self.individuals = self.annotations.individual.unique()
        self.image_paths = self.annotations.groupby("individual").apply(lambda x: x.set_index('tile_year').image_path.to_dict())
        self.image_dict = image_dict
        if train:
            self.labels = self.annotations.set_index("individual").label.to_dict()
        
        # Create augmentor
        self.transformer = augmentation.augment(image_size=self.image_size, train=train, pad_or_resize=config["pad_or_resize"])
                     
        # Pin data to memory if desired 
        if self.config["preload_images"]:
            if self.image_dict is None:
                self.image_dict = { }
                for individual in self.individuals:
                    images = { }
                    ind_annotations = self.image_paths[individual]
                    for year in self.years:
                        try:
                            year_annotations = ind_annotations[year]
                        except KeyError:
                            images[str(year)] = torch.zeros(self.config["bands"], self.image_size, self.image_size)  
                            continue
                        image_path = os.path.join(self.config["crop_dir"], year_annotations)
                        try:
                            image = utils.load_image(image_path)
                        except ValueError:
                            image = torch.zeros(self.config["bands"], self.image_size, self.image_size)                        
                        images[str(year)] = image
                    self.image_dict[individual] = images
            
    def __len__(self):
        # 0th based index
        return len(self.individuals)

    def __getitem__(self, index):
        inputs = {}
        individual = self.individuals[index]      
        if self.config["preload_images"]:
            images = self.image_dict[individual]
        else:
            images = { }
            ind_annotations = self.image_paths[individual]
            for year in self.years:
                try:
                    year_annotations = ind_annotations[year]
                except KeyError:
                    images[str(year)] = torch.zeros(self.config["bands"], self.image_size, self.image_size)  
                    continue
                image_path = os.path.join(self.config["crop_dir"], year_annotations)
                try:
                    image = utils.load_image(image_path)
                except ValueError:
                    image = torch.zeros(self.config["bands"], self.image_size, self.image_size)

                images[str(year)] = image
                
        images = {key: self.transformer(value) for key, value in images.items()}
        inputs["HSI"] = images

        if self.train:
            label = self.labels[individual]
            label = torch.tensor(label, dtype=torch.long)

            return individual, inputs, label
        else:
            return individual, inputs