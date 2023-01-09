#Prepare semi_supervised data
import glob
import torch
import copy
import torch
import random
import numpy as np
import pandas as pd
import geopandas as gpd
from pytorch_lightning import Trainer

from src.data import TreeDataset
from src.models import baseline

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
    
def select_samples(predicted_samples, config):
    """Given a unlabeled dataframe, select which samples to include in dataloader"""
    samples_to_keep = predicted_samples[predicted_samples.score > config["semi_supervised"]["threshold"]]
    
    #Optionally balance and limit
    individuals_to_keep = samples_to_keep.groupby("taxonID").apply(lambda x: x.drop_duplicates(subset="individual").sample(n=config["semi_supervised"]["max_samples_per_class"])).individual.values
    predicted_samples = predicted_samples[predicted_samples.individual.isin(individuals_to_keep)]

    return predicted_samples

def predict_unlabeled(config, annotation_df, label_to_taxon_id, m=None):
    """Predict unlabaled data with model in memory or loaded from file
    Args:
        config: a DeepTreeAttention config
        annotation_df: a pandas dataframe with image_path, to be into data.TreeDataset
    Returns:
        ensemble_df: ensembled dataframe of predictions
    """
    
    if annotation_df.empty:
        raise ValueError("Annoatation Dataframe has no rows")
    
    new_config = copy.deepcopy(config)
    new_config["crop_dir"] = new_config["semi_supervised"]["crop_dir"]
    new_config["preload_images"] = False
    new_config["workers"] = 5
    
    if m is None:
        m = baseline.TreeModel.load_from_checkpoint(new_config["semi_supervised"]["model_path"], config=new_config)
    
    trainer = Trainer(gpus=new_config["gpus"], logger=False, enable_checkpointing=False)
    ds = TreeDataset(df = annotation_df, train=False, config=new_config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=new_config["predict_batch_size"],
        shuffle=False,
        num_workers=new_config["workers"],
    )
    
    predictions = trainer.predict(m, dataloaders=data_loader)  
    
    if predictions is None:
        raise ValueError("No predictions made from the trainer dataloader")
    
    predictions = np.vstack(predictions)
    annotation_df["label"] = np.argmax(predictions,axis=1)
    annotation_df["score"] = np.max(predictions,axis=1) 
    annotation_df["taxonID"] = annotation_df.label.apply(lambda x: label_to_taxon_id[x])
    
    return annotation_df

def create_dataframe(config, label_to_taxon_id, unlabeled_df=None, m=None, client=None):
    """Generate a pytorch dataloader from unlabeled crop data"""
    
    if unlabeled_df is None:
        unlabeled_df = load_unlabeled_data(config, client=client)
        
    predicted_samples = predict_unlabeled(config, unlabeled_df, label_to_taxon_id=label_to_taxon_id, m=m)
    
    # Predict labels for each crop
    selected_df = select_samples(
        predicted_samples=predicted_samples,
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
