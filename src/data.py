#Ligthning data module
import glob
import os
from pytorch_lightning import LightningDataModule
import utils
import generate
from . import __file__
import dataset
import start_cluster
import torch

def filter_data():
    """Transform raw NEON data into clean shapefile    
    """
    pass

def split_train_test(df):
    """Split processed shapefile into train and test
    Args:
        df: pandas dataf
    Returns:
        train: geopandas frame of points
        test: geopandas frame of points
    """
    
    pass

class TreeData(LightningDataModule):
    def __init__(self):
        super().__init__()
        self.config = utils.read_config()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))
        self.data_dir = "{}/data/".format(self.ROOT)
    
    def setup(self, regenerate = False):
        #Clean data from raw csv, regenerate from scratch or check for progress and complete
        if regenerate:
            client = start_cluster.start(cpus=30)
            df = filter_data("{}/neon_vst_2021.csv".format(self.data_dir))
            train, test = split_train_test(df)   
            
            test.to_file("{}/processed/test_points.shp".format(self.data_dir))
            train.to_file("{}/processed/train_points.shp".format(self.data_dir))
            
            generate.points_to_crowns(
                field_data="{}/processed/test_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
                        
            generate.points_to_crowns(
                field_data="{}/processed/train_points.shp".format(self.data_dir),
                rgb_dir=self.config["rgb_sensor_pool"],
                savedir=self.config["crown_dir"],
                raw_box_savedir=self.config["crown_dir"],        
            )
            
            generate.generate_crops(train_crowns, savedir=self.config["crop_dir"])    
            generate.generate_crops(test_crowns, savedir=self.config["crop_dir"])  
            client.close()
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
