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

from src.models import multi_stage

def read_and_sample(path, frac):
    """Read a shapefile and sample a portion of the individuals"""
    gdf = gpd.read_file(path)
    inds = gdf.sample(frac=frac).individual.unique()
    gdf = gdf[gdf.individual.isin(inds)]
    
    return gdf
    
def load_unlabeled_data(config, client=None):
    semi_supervised_crops_csvs = glob.glob("{}/*.shp".format(config["semi_supervised"]["crop_dir"]))
    
    if len(semi_supervised_crops_csvs) == 0:
        raise ValueError("No .shp files found in {}".format(config["semi_supervised"]["crop_dir"]))
    
    random.shuffle(semi_supervised_crops_csvs)
    semi_supervised_crops_csvs = semi_supervised_crops_csvs[:config["semi_supervised"]["limit_shapefiles"]]
    if client:
        semi_supervised_crops = client.map(read_and_sample, semi_supervised_crops_csvs, frac=config["semi_supervised"]["sample_fraction_of_individuals"])
        semi_supervised_crops = client.gather(semi_supervised_crops)
    else:
        semi_supervised_crops = [read_and_sample(x, frac=config["semi_supervised"]["sample_fraction_of_individuals"]) for x in semi_supervised_crops_csvs]
        
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
    individuals_to_keep = samples_to_keep.groupby("taxonID").apply(lambda x: x.drop_duplicates(subset="individual").sample(frac=1).head(n=config["semi_supervised"]["max_samples_per_class"])).individual.values
    predicted_samples = predicted_samples[predicted_samples.individual.isin(individuals_to_keep)].reset_index(drop=True)

    return predicted_samples

def predict_unlabeled(config, annotation_df, m=None):
    """Predict unlabaled data with model in memory or loaded from file
    Args:
        config: a DeepTreeAttention config
        annotation_df: a pandas dataframe with image_path, to be into TreeDataset
    Returns:
        ensemble_df: ensembled dataframe of predictions
    """
    
    if annotation_df.empty:
        raise ValueError("Annoatation Dataframe has no rows")
    
    new_config = copy.deepcopy(config)
    new_config["crop_dir"] = new_config["semi_supervised"]["crop_dir"]
    new_config["workers"] = new_config["semi_supervised"]["workers"]
    new_config["preload_images"] = new_config["semi_supervised"]["preload_images"]
    
    if m is None:
        m = multi_stage.MultiStage.load_from_checkpoint(new_config["semi_supervised"]["model_path"], config=new_config)
    
    trainer = Trainer(gpus=new_config["gpus"], logger=False, enable_checkpointing=False)
    ds = multi_stage.TreeDataset(df = annotation_df, train=False, config=new_config)
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
    results = m.gather_predictions(predictions)
    ensemble_df = m.ensemble(results)
    output = annotation_df.merge(ensemble_df, on="individual")
    output["taxonID"] = output["ensembleTaxonID"]
    output["score"] = output["ens_score"]
    
    return output

def create_dataframe(config, unlabeled_df=None, m=None, client=None):
    """Generate a pytorch dataloader from unlabeled crop data"""
    
    if unlabeled_df is None:
        unlabeled_df = load_unlabeled_data(config, client=client)
        
    predicted_samples = predict_unlabeled(config, unlabeled_df, m=m)
    
    # Predict labels for each crop
    selected_df = select_samples(
        predicted_samples=predicted_samples,
        config=config
    )
    
    return selected_df
