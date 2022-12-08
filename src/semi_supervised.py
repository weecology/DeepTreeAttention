#Prepare semi_supervised data
import copy
import glob
import torch
from src.models import multi_stage
from src.data import TreeDataset
from pytorch_lightning import Trainer
import pandas as pd

def load_unlabeled_data(config):
    semi_supervised_crops_csvs = glob.glob("{}/*.csv".format(config["semi_supervised"]["crop_dir"]))
    semi_supervised_crops = pd.concat([pd.read_csv(x) for x in semi_supervised_crops_csvs])
    if config["semi_supervised"]["site_filter"] is None:
        site_semi_supervised_crops = semi_supervised_crops[semi_supervised_crops.image_path.str.contains(config["semi_supervised"]["site_filter"])]
    else:
        site_semi_supervised_crops = semi_supervised_crops
    
    site_semi_supervised_crops = site_semi_supervised_crops.head(config["semi_supervised"]["num_samples"])
    
    return site_semi_supervised_crops
    
def predict_unlabeled(config, annotation_df, m=None):
    """Predict unlabaled data with model in memory or loaded from file
    Args:
        config: a DeepTreeAttention config
        annotation_df: a pandas dataframe with image_path, to be into data.TreeDataset
    Returns:
        ensemble_df: ensembled dataframe of predictions
    """
    config = copy.deepcopy(config)
    config["crop_dir"] = config["semi_supervised"]["crop_dir"]
        
    if m is None:
        m = multi_stage.MultiStage.load_from_checkpoint(config["semi_supervised"]["model_path"], config=config)
    
    trainer = Trainer(gpus=config["gpus"], logger=False, enable_checkpointing=False)
    ds = TreeDataset(annotation_df, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)    
    ensemble_df = m.ensemble(results)
    
    ensemble_df["taxonID"] = ensemble_df.ensembleTaxonID
    ensemble_df["label"] = ensemble_df.taxonID.apply(lambda x: m.species_label_dict[x])
    
    return ensemble_df

def select_samples(unlabeled_df, ensemble_df, config):
    """Given a unlabeled dataframe, select which samples to include in dataloader"""
    samples_to_keep = ensemble_df[ensemble_df.ens_score > config["semi_supervised"]["threshold"]].individual
    unlabeled_df = unlabeled_df[unlabeled_df.individual.isin(samples_to_keep)]
    
    return unlabeled_df
        
def create_dataloader(config):
    """Generate a pytorch dataloader from unlabeled crop data"""
    site_semi_supervised_crops = load_unlabeled_data(config)
    ensemble_df = predict_unlabeled(config, site_semi_supervised_crops)
    
    #Predict labels for each crop
    unlabeled_df = select_samples(
        unlabeled_df=site_semi_supervised_crops,
        ensemble_df=ensemble_df,
        config=config
    )
    
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

