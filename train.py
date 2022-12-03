#Train
import comet_ml
import glob
import geopandas as gpd
import os
import numpy as np
from src import data
from src import start_cluster
from src.models import baseline, Hang2020
from src import visualize
from src import metrics
import subprocess
import sys
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.profiler import AdvancedProfiler

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
    
    if client:
        client.close()
    
    comet_logger.experiment.log_parameter("train_hash",hash_pandas_object(data_module.train))
    comet_logger.experiment.log_parameter("test_hash",hash_pandas_object(data_module.test))
    comet_logger.experiment.log_parameter("num_species",data_module.num_classes)
    comet_logger.experiment.log_table("train.csv", data_module.train)
    comet_logger.experiment.log_table("test.csv", data_module.test)
    
    if not config["use_data_commit"]:
        comet_logger.experiment.log_table("novel_species.csv", data_module.novel)
    
    train = data_module.train.copy()
    test = data_module.test.copy()
    crowns = data_module.crowns.copy()
    
    #remove graves
    train = train[~train.individual.str.contains("graves")].reset_index(drop=True)
    test = test[~test.individual.str.contains("graves")].reset_index(drop=True)
    
    model = Hang2020.vanilla_CNN(bands=config["bands"], classes=data_module.num_classes)
            
    #Loss weight, balanced
    loss_weight = []
    for x in data_module.species_label_dict:
        loss_weight.append(1/data_module.train[data_module.train.taxonID==x].shape[0])
        
    loss_weight = np.array(loss_weight/np.max(loss_weight))
    #Provide min value
    loss_weight[loss_weight < 0.5] = 0.5  
    
    comet_logger.experiment.log_parameter("loss_weight", loss_weight)
    
    m = baseline.TreeModel(
        model=model, 
        config=config
        classes=data_module.num_classes, 
        loss_weight=loss_weight,
        label_dict=data_module.species_label_dict)
        
    #Create trainer
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = Trainer(
        gpus=data_module.config["gpus"],
        fast_dev_run=data_module.config["fast_dev_run"],
        max_epochs=data_module.config["epochs"],
        accelerator=data_module.config["accelerator"],
        num_sanity_val_steps=0,
        val_check_interval=10,
        enable_checkpointing=False,
        callbacks=[lr_monitor],
        logger=comet_logger)
    
    trainer.fit(m)
    
    #Save model checkpoint
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pt".format(comet_logger.experiment.id))
    
    # Prediction datasets are indexed by year, but full data is given to each model before ensembling
    print("Before prediction, the taxonID value counts")
    print(test.taxonID.value_counts())
    results = m.evaluate_crowns(
        data_module.val_dataloader(),
        crowns = data_module.crowns,
        experiment=comet_logger.experiment,
    )
    rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
    
    #Visualizations
    visualize.plot_spectra(results, crop_dir=config["crop_dir"], experiment=comet_logger.experiment)
    visualize.rgb_plots(
        df=results,
        config=config,
        test_crowns=data_module.crowns,
        test_points=data_module.canopy_points,
        plot_n_individuals=config["plot_n_individuals"],
        experiment=comet_logger.experiment)
    visualize.confusion_matrix(
        comet_experiment=comet_logger.experiment,
        results=results,
        species_label_dict=data_module.species_label_dict,
        test_crowns=data_module.crowns,
        test=data_module.test,
        test_points=data_module.canopy_points,
        rgb_pool=rgb_pool
    )
    
    #Log prediction
    comet_logger.experiment.log_table("test_predictions.csv", results)
    
    #Within site confusion
    site_lists = data_module.train.groupby("label").site.unique()
    within_site_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=site_lists)
    comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)
    
    #Within plot confusion
    plot_lists = data_module.train.groupby("label").plotID.unique()
    within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=plot_lists)
    comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)
    
if __name__ == "__main__":
    main()