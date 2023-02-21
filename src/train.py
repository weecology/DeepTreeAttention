# Train
import comet_ml
import os
import numpy as np
from pandas.util import hash_pandas_object
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src import data, start_cluster
from src.models import multi_stage, Hang2020, baseline
from src.data import __file__

ROOT = os.path.dirname(os.path.dirname(__file__))

def main(config, site=None, git_branch=None, git_commit=None):
    #Create datamodule
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")     
    
    #Generate new data or use previous run
    if config["use_data_commit"]:
        config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
        client = None    
    else:
        crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
        os.mkdir(crop_dir)
        if not git_branch == "pytest":
            client = start_cluster.start(cpus=100, mem_size="4GB")   
        else:
            client = None
        config["crop_dir"] = crop_dir
    
    comet_logger.experiment.log_parameter("git branch",git_branch)
    if site:
        tag = "{}_{}".format(git_branch, site)
    else:
        tag = git_branch
    comet_logger.experiment.add_tag(tag)
    comet_logger.experiment.log_parameter("commit hash",git_commit)
    comet_logger.experiment.log_parameters(config)
    
    data_module = data.TreeData(
        csv_file="{}/data/raw/neon_vst_data_2022.csv".format(ROOT),
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        site=site,
        comet_logger=comet_logger)
    
    if config["create_pretrain_model"]:
        config["existing_test_csv"] = "{}/test_{}.csv".format(data_module.data_dir, data_module.experiment_id)
        config["pretrain_state_dict"] = pretrain_model(comet_logger, config)
    if client:
        client.close()
    
    comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    comet_logger.experiment.log_table("test.csv", data_module.test)
    
    #always assert that there is no train in test, skip for debug
    if not git_branch == "pytest":
        assert data_module.train[data_module.train.individual.isin(data_module.test.individual)].empty
    
    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)
    
    m = multi_stage.MultiStage(
        train_df=data_module.train, 
        test_df=data_module.test, 
        config=data_module.config)          
        
    comet_logger = train_model(data_module, comet_logger, m, site)
    
    return comet_logger

def pretrain_model(comet_logger, config, client=None):
    """Pretain a model with samples from other sites
    Args:
        comet_logger: a cometML logger
    Returns:
        path: a path on disk for trained model state dict
    """
    with comet_logger.experiment.context_manager("pretrain"):
        pretrain_module = data.TreeData(
            csv_file="{}/data/raw/neon_vst_data_2022.csv".format(ROOT),
            data_dir=config["crop_dir"],
            config=config,
            client=client,
            site="all",
            filter_species_site=None,
            comet_logger=comet_logger)
        
        model = Hang2020.Single_Spectral_Model(bands=pretrain_module.config["bands"], classes=pretrain_module.num_classes)
        m = baseline.TreeModel(
            model=model,
            classes=pretrain_module.num_classes,
            label_dict=pretrain_module.species_label_dict,
            config=pretrain_module.config) 
        
        path = "{}/{}_state_dict.pt".format(config["snapshot_dir"], comet_logger.experiment.id)
        torch.save(m.model.state_dict(), path) 
        trainer = Trainer(
            gpus=pretrain_module.config["gpus"],
            fast_dev_run=pretrain_module.config["fast_dev_run"],
            max_epochs=pretrain_module.config["epochs"],
            accelerator=pretrain_module.config["accelerator"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=pretrain_module.config["validation_interval"],
            enable_checkpointing=False,
            logger=comet_logger)
        
        trainer.fit(m, datamodule=pretrain_module)
        
        return path
        
def train_model(data_module, comet_logger, m, name):
    """Model training loop"""
    print(name)
    print(data_module.species_label_dict)    
    comet_logger.experiment.log_parameter("site", name)           

    for key, value in m.test_dataframes.items():
        comet_logger.experiment.log_table("test_{}.csv".format(key), value)

    for key, value in m.train_dataframes.items():
        comet_logger.experiment.log_table("train_{}.csv".format(key), value)
    
    for key, level_label_dict in m.level_label_dicts.items():
        print("Label dict for {} is {}".format(key, level_label_dict))
        
    comet_logger.experiment.log_parameters(data_module.train.taxonID.value_counts().to_dict(), prefix="count")
    
    #Create trainer
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
        
    for key in m.level_names:
        trainer = Trainer(
            gpus=data_module.config["gpus"],
            fast_dev_run=data_module.config["fast_dev_run"],
            max_epochs=data_module.config["epochs"],
            accelerator=data_module.config["accelerator"],
            num_sanity_val_steps=0,
            check_val_every_n_epoch=data_module.config["validation_interval"],
            callbacks=[lr_monitor],
            enable_checkpointing=False,
            logger=comet_logger)
        
        m.current_level = key
        m.configure_optimizers()
        trainer.fit(m)
        
    #Save model checkpoint
    if data_module.config["snapshot_dir"] is not None:
        trainer.save_checkpoint("{}/{}_{}.pt".format(data_module.config["snapshot_dir"], comet_logger.experiment.id, name))
    
    ds = multi_stage.TreeDataset(df=data_module.test, train=False, config=data_module.config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results = results.merge(data_module.test[["individual","taxonID","label","siteID"]], on="individual")
    comet_logger.experiment.log_table("nested_predictions.csv", results)
    
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=comet_logger.experiment
    )
    
    comet_logger.experiment.log_confusion_matrix(
        ensemble_df.label.values,
        ensemble_df.ens_label.values,
        labels=list(data_module.species_label_dict.keys()),
        max_categories=len(data_module.species_label_dict.keys()),
        title=name,
        file_name="{}.json".format(name)
    )
    
    # Log prediction
    comet_logger.experiment.log_table("test_predictions.csv", ensemble_df)
    
    return m
