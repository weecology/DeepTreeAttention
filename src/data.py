#Ligthning data module
import os
from pytorch_lightning import LightningDataModule
from src import utils
from src import __file__
from src import dataset
import torch

def filter_data():
    """Transform raw NEON data into clean shapefile    
    """
    pass

def split_train_test():
    """Split processed shapefile into train and test"""
    pass

class TreeData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.config = utils.read_config()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = "{}/data/".format(self.ROOT)
    
    def setup(self, regenerate = False):
        #Clean data from raw csv
        if regenerate:
            filter_data()
        else:
            if not os.path.exists("{}/processed/filtered_data.csv".format(self.data_dir)):
                filter_data()
        #Split train test
        if regenerate:
            split_train_test()
        else:
            if not os.path.exists("{}/processed/train.csv".format(self.data_dir)):
                split_train_test()

    def train_dataloader(self):
        ds = dataset.TreeDataset(csv_file = "{}/processed/train.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def val_dataloader(self):
        ds = dataset.TreeDataset(csv_file = "{}/processed/test.csv".format(self.ROOT))
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return data_loader
