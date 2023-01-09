#Train
import os
#os.environ['COMET_LOGGING_FILE_LEVEL'] = 'DEBUG'
#os.environ['COMET_LOGGING_FILE'] = './comet.log'

import comet_ml
import copy
import glob
import geopandas as gpd
import os
import numpy as np
from src import data
from src import start_cluster
from src.models import joint_semi, Hang2020
from src import visualize, semi_supervised, metrics
import subprocess
import sys
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import SimpleProfiler

import pandas as pd
from pandas.util import hash_pandas_object

def main():
    #Get branch name for the comet tag
    git_branch=sys.argv[1]
    git_commit=sys.argv[2]
    
    #Create datamodule
    config = data.read_config("config.yml")
    comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")    
    
    #Generate new data or use previous run
    if config["use_data_commit"]:
        config["crop_dir"] = os.path.join(config["data_dir"], config["use_data_commit"])
        client = None    
    else:
        crop_dir = os.path.join(config["data_dir"], comet_logger.experiment.get_key())
        os.mkdir(crop_dir)
        client = start_cluster.start(cpus=100, mem_size="4GB")    
        config["crop_dir"] = crop_dir
    
    comet_logger.experiment.log_parameter("git branch",git_branch)
    comet_logger.experiment.add_tag(git_branch)
    comet_logger.experiment.log_parameter("commit hash",git_commit)
    comet_logger.experiment.log_parameters(config)
    
    data_module = data.TreeData(
        csv_file="data/raw/neon_vst_data_2022.csv",
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        metadata=True,
        comet_logger=comet_logger)
    
    supervised_train = data_module.train.copy()    
    
    #Overwrite train with the semi-supervised crops
    if data_module.config["semi_supervised"]["semi_supervised_train"] is None:
        client = start_cluster.start(cpus=10, mem_size="6GB")   
    else:
        client = None
    
    comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
    comet_logger.experiment.log_table("test.csv", data_module.test)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    
    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)
    
    test = data_module.test.copy()
    model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=data_module.num_classes)
    
    ###Loss weight, balanced
    loss_weight = []
    for x in data_module.species_label_dict:
        count_in_df = data_module.train[data_module.train.taxonID==x].shape[0]
        if count_in_df == 0:
            loss_weight.append(0)
        else:
            loss_weight.append(1/count_in_df)
                        
    loss_weight = np.array(loss_weight/np.max(loss_weight))
    
    #Just one year
    test = data_module.test
    train = data_module.train
    
    #test = test[test.tile_year=="2021"]
    #train = train[train.tile_year=="2021"]
    
    m = joint_semi.TreeModel(
        model=model, 
        config=config,
        client=client,
        classes=data_module.num_classes, 
        loss_weight=loss_weight,
        supervised_test=test,
        supervised_train=train,
        label_dict=data_module.species_label_dict)
    
    comet_logger.experiment.log_table("semi_supervised_train.csv", m.semi_supervised_train)
    m.semi_supervised_train.to_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/semi_supervised/{}.csv".format(comet_logger.experiment.id))
    
    #Create trainer
    profiler=SimpleProfiler("/blue/ewhite/b.weinstein/DeepTreeAttention/logs/{}_profiler.out".format(comet_logger.experiment.id))
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        gpus=data_module.config["gpus"],
        fast_dev_run=data_module.config["fast_dev_run"],
        max_epochs=data_module.config["epochs"],
        accelerator=data_module.config["accelerator"],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=data_module.config["validation_interval"],
        callbacks=[lr_monitor],
        enable_checkpointing=False,
        profiler=profiler,
        logger=comet_logger)
    
    trainer.fit(m)

    #Save model checkpoint and profile
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pt".format(comet_logger.experiment.id))
    torch.save(m.model.state_dict(), "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_state_dict.pt".format(comet_logger.experiment.id))
    comet_logger.experiment.log_asset("profile","/blue/ewhite/b.weinstein/DeepTreeAttention/logs/{}_profiler.out".format(comet_logger.experiment.id))
    
    # Prediction datasets are indexed by year, but full data is given to each model before ensembling
    results = m.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment        
    )

    rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)    
    #Create a per-site confusion matrix by recoding each site as a seperate set of labels
    for site in results.siteID.unique():
        site_result = results[results.siteID==site]
        combined_species = np.unique(site_result[['taxonID',"pred_taxa_top1"]].values)
        site_labels = {value:key for key, value in enumerate(combined_species)}
        y = [site_labels[x] for x in site_result.taxonID.values]
        ypred = [site_labels[x] for x in site_result.pred_taxa_top1.values]
        taxonlabels = [key for key, value in site_labels.items()]
        comet_logger.experiment.log_confusion_matrix(
            y,
            ypred,
            labels=taxonlabels,
            max_categories=len(taxonlabels),
            file_name="{}.json".format(site),
            title="{}".format(site)
        )
    
    #Log prediction
    comet_logger.experiment.log_table("test_predictions.csv", results)
    
    #Within site confusion
    site_lists = data_module.train.groupby("label").siteID.unique()
    within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=site_lists)
    comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)
    
    #Within plot confusion
    plot_lists = data_module.train.groupby("label").plotID.unique()
    within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=plot_lists)
    comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)

if __name__ == "__main__":
    main()
