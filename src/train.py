# Train
import comet_ml
import os
import gc
import numpy as np
from pandas.util import hash_pandas_object
import pandas as pd
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src import data
from src.models import baseline, year
from src.data import __file__
from src import visualize

ROOT = os.path.dirname(os.path.dirname(__file__))

def main(config, git_branch=None, git_commit=None, client=None, csv_file="{}/data/raw/neon_vst_data_2023.csv".format(ROOT)):
    #Create datamodule
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")     
    
    #Generate new data or use previous run
    if config["use_data_commit"]:
        config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
        client = None    
    else:
        crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
        os.mkdir(crop_dir)
        config["crop_dir"] = crop_dir
    
    comet_logger.experiment.log_parameter("git branch",git_branch)
    tag = git_branch
    comet_logger.experiment.add_tag(tag)
    comet_logger.experiment.log_parameter("commit hash",git_commit)
    comet_logger.experiment.log_parameters(config)
    
    #If train test split does not exist create one
    if not os.path.exists("{}/train_{}.csv".format(config["crop_dir"], config["train_test_commit"])):
        print("Create new train test split")
        create_train_test = True
    else:
        create_train_test = False
        
    data_module = data.TreeData(
        csv_file=csv_file,
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        create_train_test=create_train_test,
        experiment_id="{}".format(comet_logger.experiment.id),
        comet_logger=comet_logger)
    
    comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    comet_logger.experiment.log_table("test.csv", data_module.test)
    
    test_species_per_site = data_module.test.groupby("siteID").apply(lambda x: len(x.taxonID.unique())).to_dict()
    for site, value in test_species_per_site.items():
        comet_logger.experiment.log_parameter("species",value)
    
    test_samples_per_site = data_module.test.groupby("individual").apply(lambda x: x.head(1)).groupby("siteID").apply(lambda x: x.shape[0])
    for site, value in test_samples_per_site.items():
        comet_logger.experiment.log_parameter("test_samples",value)
    
    train_samples_per_site = data_module.train.groupby("individual").apply(lambda x: x.head(1)).groupby("siteID").apply(lambda x: x.shape[0])
    for site, value in train_samples_per_site.items():
        comet_logger.experiment.log_parameter("train_samples",value)
        
    #always assert that there is no train in test, skip for debug
    if not git_branch == "pytest":
        assert data_module.train[data_module.train.individual.isin(data_module.test.individual)].empty
    
    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)
    
    years = data_module.train.tile_year.unique()
    year_model = year.learned_ensemble(years=years, classes=data_module.num_classes, config=config)

    m = baseline.TreeModel(
        model=year_model,
        classes=data_module.num_classes,
        label_dict = data_module.species_label_dict,
        loss_weight=torch.ones(data_module.num_classes),
        config=config
        )

    comet_logger = train_model(data_module, comet_logger, m)
    
    return comet_logger
        
def train_model(data_module, comet_logger, m):
    """Model training loop"""
    print(data_module.species_label_dict)    

    comet_logger.experiment.log_parameters(data_module.train.taxonID.value_counts().to_dict(), prefix="count")
    
    #Create trainer
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
    trainer = Trainer(
        fast_dev_run=data_module.config["fast_dev_run"],
        max_epochs=data_module.config["epochs"],
        accelerator=data_module.config["accelerator"],
        check_val_every_n_epoch=data_module.config["validation_interval"],
        callbacks=[lr_monitor],
        enable_progress_bar=False,
        enable_checkpointing=False,
        devices=data_module.config["gpus"],
        num_sanity_val_steps=0,
        logger=comet_logger)

    trainer.fit(m, datamodule=data_module)
        
    #Save model checkpoint
    if data_module.config["snapshot_dir"] is not None:
        trainer.save_checkpoint("{}/{}_{}.pt".format(data_module.config["snapshot_dir"], comet_logger.experiment.id, name))
    
    ds = data.TreeDataset(df=data_module.test, train=False, config=data_module.config)    
    predictions = trainer.predict(m, dataloaders=data_module.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results = results.merge(data_module.test[["individual","taxonID","label","siteID","RGB_tile"]], on="individual")
    
    comet_logger.experiment.log_table("nested_predictions.csv", results)
    
    ensemble_df = m.evaluation_scores(
        results,
        experiment=comet_logger.experiment
    )
    
    for site in ensemble_df.siteID.unique():
        site_df = ensemble_df[ensemble_df.siteID==site]
        visualize.confusion_matrix(
            comet_experiment=comet_logger.experiment,
            yhats=site_df.yhat.values,
            y=site_df.label.values,
            individuals=site_df.individual.values,
            RGB_tiles=site_df.RGB_tile.values,
            labels=list(data_module.species_label_dict.keys()),
            test_points=data_module.canopy_points,
            test_crowns=data_module.crowns,
            name=site
        )
        
    # Log prediction
    comet_logger.experiment.log_table("test_predictions.csv", ensemble_df)
    
    return m
