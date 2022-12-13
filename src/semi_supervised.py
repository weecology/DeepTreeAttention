#Prepare semi_supervised data
import copy
import glob
import torch
from src.models import multi_stage
from src.data import TreeDataset
from pytorch_lightning import Trainer
import pandas as pd
import geopandas as gpd
import torch
import numpy as np
import random

def load_unlabeled_data(config):
    semi_supervised_crops_csvs = glob.glob("{}/*.shp".format(config["semi_supervised"]["crop_dir"]))
    random.shuffle(semi_supervised_crops_csvs)
    semi_supervised_crops = pd.concat([gpd.read_file(x) for x in semi_supervised_crops_csvs[:config["semi_supervised"]["limit_shapefiles"]]])
    if config["semi_supervised"]["site_filter"] is None:
        site_semi_supervised_crops = semi_supervised_crops[semi_supervised_crops.image_path.str.contains(config["semi_supervised"]["site_filter"])]
    else:
        site_semi_supervised_crops = semi_supervised_crops
    
    site_semi_supervised_crops = site_semi_supervised_crops.head(config["semi_supervised"]["num_samples"])
    
    return site_semi_supervised_crops
    
def predict_unlabeled(config, annotation_df, label_to_taxon_id, m=None):
    """Predict unlabaled data with model in memory or loaded from file
    Args:
        config: a DeepTreeAttention config
        annotation_df: a pandas dataframe with image_path, to be into data.TreeDataset
    Returns:
        ensemble_df: ensembled dataframe of predictions
    """
    new_config = copy.deepcopy(config)
    new_config["crop_dir"] = new_config["semi_supervised"]["crop_dir"]
        
    if m is None:
        m = multi_stage.MultiStage.load_from_checkpoint(new_config["semi_supervised"]["model_path"], config=new_config)
    
    trainer = Trainer(gpus=new_config["gpus"], logger=False, enable_checkpointing=False)
    ds = TreeDataset(df = annotation_df, train=False, config=new_config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=new_config["predict_batch_size"],
        shuffle=False,
        num_workers=new_config["workers"],
    )
    
    predictions = trainer.predict(m, dataloaders=data_loader)  
    annotation_df["label"] = np.argmax(np.concatenate(predictions, 1),axis=1)
    annotation_df["score"] = np.max(np.concatenate(predictions, 1),axis=1) 
    annotation_df["taxonID"] = annotation_df.label.apply(lambda x: label_to_taxon_id[x])
    
    return annotation_df

def select_samples(unlabeled_df, ensemble_df, config):
    """Given a unlabeled dataframe, select which samples to include in dataloader"""
    samples_to_keep = ensemble_df[ensemble_df.score > config["semi_supervised"]["threshold"]].individual
    unlabeled_df = unlabeled_df[unlabeled_df.individual.isin(samples_to_keep)]
    
    return unlabeled_df
        
def create_dataframe(config, label_to_taxon_id, unlabeled_df=None, m=None):
    """Generate a pytorch dataloader from unlabeled crop data"""
    
    if unlabeled_df is None:
        unlabeled_df = load_unlabeled_data(config)
        
    ensemble_df = predict_unlabeled(config, unlabeled_df, label_to_taxon_id=label_to_taxon_id, m=m)
    
    # Predict labels for each crop
    selected_df = select_samples(
        unlabeled_df=unlabeled_df,
        ensemble_df=ensemble_df,
        config=config
    )
    
    return selected_df


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