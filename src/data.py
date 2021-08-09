#Ligthning data module
import glob
import os
from pytorch_lightning import LightningDataModule
from src import generate
from . import __file__
from src import dataset 
from src import start_cluster
import torch
import pandas as pd
import yaml

def filter_data(path, min_samples=None, filter_CHM=True):
    """Transform raw NEON data into clean shapefile   
    Args:
        min_samples: each class must have x samples
    """
    df = pd.read_csv(path)
    
    if min_samples:
        keep_taxa = df.taxonID.value_counts()
        keep_taxa[keep_taxa > min_samples]
    
    df = df[df.taxonID.isin(keep_taxa)]
    
    if filter_CHM:
        pass
    
    return df


def split_train_test(path, min_resample):
    """Split processed shapefile into train and test
    Args:
        df: pandas datadf
        min_resample: classes will be sample to have atleast n samples
    Returns:
        train: geopandas frame of points
        test: geopandas frame of points
    """
    
    df[df.taxonID == ""]
    
    return df


def read_config(config_path):
    """Read config yaml file"""
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))

    return config

class TreeData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = "{}/data/".format(self.ROOT)
        self.config = read_config("{}/config.yml".format(self.ROOT))        
    
    def setup(self, csv_file, regenerate = False):
        #Clean data from raw csv, regenerate from scratch or check for progress and complete
        if regenerate:
            #client = start_cluster.start(cpus=30)
            df = filter_data(csv_file, min_samples=self.config["min_samples"], filter_CHM=self.config["filter_CHM"])
            train, test = split_train_test(df, min_resample = self.config["min_resample"])   
            
            test.to_file("{}/processed/test_points.shp".format(self.data_dir))
            train.to_file("{}/processed/train_points.shp".format(self.data_dir))
            
            generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["validation"]["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
                        
            generate.points_to_crowns(
                field_data="{}/processed/train_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["train"]["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
            
            #For each shapefile, create crops and csv file
            train_crops = []
            for x in glob.glob("*.shp".format(self.config["train"]["crown_dir"])):
                crop_df = generate.generate_crops(x, savedir=self.config["crop_dir"])
                train_crops.append(crop_df)
                
            test_crops = []
            for x in glob.glob("*.shp".format(self.config["validation"]["crown_dir"])):
                crop_df = generate.generate_crops(x, savedir=self.config["crop_dir"])
                test_crops.append(crop_df)                
        if not os.path.exists("{}/processed/filtered_data.csv".format(self.data_dir)):
            filter_data()
        if not os.path.exists("{}/processed/train_points.shp".format(self.data_dir)):
            split_train_test()
        if not os.path.exists("{}/processed/train_crowns.shp".format(self.data_dir)):
            #test data
            generate.points_to_crowns(
                field_data="{}/processed/train_points.csv".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
                        
            generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
            
        if len(glob.glob(self.config["crop_dir"])) == 0:
            generate.generate_crops(savedir=self.config["crop_dir"])            

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
