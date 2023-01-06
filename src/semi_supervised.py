#Prepare semi_supervised data
import copy
import glob
import torch
from src.models import baseline
from src.data import TreeDataset
from pytorch_lightning import Trainer
import pandas as pd
import geopandas as gpd
import torch
import numpy as np
import random

def load_unlabeled_data(config, client=None):
    semi_supervised_crops_csvs = glob.glob("{}/*.shp".format(config["semi_supervised"]["crop_dir"]))
    
    if len(semi_supervised_crops_csvs) == 0:
        raise ValueError("No .shp files found in {}".format(config["semi_supervised"]["crop_dir"]))
    
    random.shuffle(semi_supervised_crops_csvs)
    semi_supervised_crops_csvs = semi_supervised_crops_csvs[:config["semi_supervised"]["limit_shapefiles"]]
    if client:
        semi_supervised_crops = client.map(gpd.read_file, semi_supervised_crops_csvs)
        semi_supervised_crops = client.gather(semi_supervised_crops)
    else:
        semi_supervised_crops = [gpd.read_file(x) for x in semi_supervised_crops_csvs]
        
    semi_supervised_crops = pd.concat(semi_supervised_crops)
    
    #if present remove dead trees
    try:
        semi_supervised_crops = semi_supervised_crops[semi_supervised_crops.dead_label==0]
    except:
        pass
    
    if config["semi_supervised"]["site_filter"] is not None:
        site_semi_supervised_crops = semi_supervised_crops[semi_supervised_crops.image_path.str.contains(config["semi_supervised"]["site_filter"])]
    else:
        site_semi_supervised_crops = semi_supervised_crops
    
    site_semi_supervised_crops = site_semi_supervised_crops.sample(frac=1)
    individuals_to_keep = pd.Series(site_semi_supervised_crops.individual.unique()).sample(n=config["semi_supervised"]["num_samples"])    
    site_semi_supervised_crops = site_semi_supervised_crops[site_semi_supervised_crops.individual.isin(individuals_to_keep)]
    
    # Reset the index to 0:n_samples
    site_semi_supervised_crops = site_semi_supervised_crops.reset_index(drop=True)
    
    # All individuals need atleast 2 years of day
    counts = site_semi_supervised_crops.individual.value_counts()
    individuals_to_keep = counts[~(counts == 1)].index.values
    site_semi_supervised_crops = site_semi_supervised_crops[site_semi_supervised_crops.individual.isin(individuals_to_keep)]
    
    return site_semi_supervised_crops
        
def create_dataframe(config, label_to_taxon_id, unlabeled_df=None, m=None, client=None):
    """Generate a pytorch dataloader from unlabeled crop data"""
    
    if unlabeled_df is None:
        unlabeled_df = load_unlabeled_data(config, client=client)
    
    return unlabeled_df

def create_dataloader(unlabeled_df, config):
    semi_supervised_ds = TreeDataset(
        df=unlabeled_df,
        train=False,
        config=config)
    
    data_loader = torch.utils.data.DataLoader(
        semi_supervised_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["workers"]
    )        
    
    return data_loader
