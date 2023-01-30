#Train
import os
import comet_ml
import copy
import glob
import geopandas as gpd
import os
import subprocess
import sys
import torch
import numpy as np
import pandas as pd

from src import data, start_cluster, visualize, semi_supervised, metrics
from src.models import joint_semi
from src.models.multi_stage import TreeDataset
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor

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
        client = start_cluster.start(cpus=20, mem_size="30GB")   
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

    #Just one year
    test = data_module.test
    train = data_module.train
    
    m = joint_semi.TreeModel(
        config=config,
        taxonomic_csv="data/raw/families.csv",
        supervised_test=test,
        supervised_train=train)
    
    comet_logger.experiment.log_table("semi_supervised_train.csv", m.semi_supervised_train)
    m.semi_supervised_train.to_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/semi_supervised/{}.csv".format(comet_logger.experiment.id))
    
    #Create trainer
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
        logger=comet_logger)
    
    trainer.fit(m)

    #Save model checkpoint and profile
    trainer.save_checkpoint("/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}.pt".format(comet_logger.experiment.id))
    torch.save(m.model.state_dict(), "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/{}_state_dict.pt".format(comet_logger.experiment.id))

    # Prediction datasets are indexed by year, but full data is given to each model before ensembling
    print("Before prediction, the taxonID value counts")
    print(test.taxonID.value_counts())
    
    ds = TreeDataset(df=test, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    results["individual"] = results["individual"]
    results_with_data = results.merge(crowns, on="individual")
    comet_logger.experiment.log_table("nested_predictions.csv", results_with_data)
    
    results = results.merge(data_module.test, on=["individual"])
    ensemble_df = m.ensemble(results)
    ensemble_df = m.evaluation_scores(
        ensemble_df,
        experiment=comet_logger.experiment
    )
    
    #Log prediction
    comet_logger.experiment.log_table("ensemble_df.csv", ensemble_df)
    
    #Visualizations
    ensemble_df["pred_taxa_top1"] = ensemble_df.ensembleTaxonID
    ensemble_df["pred_label_top1"] = ensemble_df.ens_label
    rgb_pool = glob.glob(data_module.config["rgb_sensor_pool"], recursive=True)
    
    #Limit to 1 individual for confusion matrix
    ensemble_df = ensemble_df.reset_index(drop=True)
    ensemble_df = ensemble_df.groupby("individual").apply(lambda x: x.head(1))
    test = test.groupby("individual").apply(lambda x: x.head(1)).reset_index(drop=True)
        
    #Create a per-site confusion matrix by recoding each site as a seperate set of labels
    for site in ensemble_df.siteID.unique():
        site_result = ensemble_df[ensemble_df.siteID==site]
        combined_species = np.unique(site_result[['taxonID', 'ensembleTaxonID']].values)
        site_labels = {value:key for key, value in enumerate(combined_species)}
        y = [site_labels[x] for x in site_result.taxonID.values]
        ypred = [site_labels[x] for x in site_result.ensembleTaxonID.values]
        taxonlabels = [key for key, value in site_labels.items()]
        comet_logger.experiment.log_confusion_matrix(
            y,
            ypred,
            labels=taxonlabels,
            max_categories=len(taxonlabels),
            file_name="{}.json".format(site),
            title=site
        )

    #Within site confusion
    site_lists = data_module.train.groupby("label").siteID.unique()
    within_site_confusion = metrics.site_confusion(y_true=ensemble_df.label, y_pred=ensemble_df.ens_label, site_lists=site_lists)
    comet_logger.experiment.log_metric("within_site_confusion", within_site_confusion)
    
    #Within plot confusion
    plot_lists = data_module.train.groupby("label").plotID.unique()
    within_plot_confusion = metrics.site_confusion(y_true=ensemble_df.label, y_pred=ensemble_df.ens_label, site_lists=plot_lists)
    comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)
    
    # Cross temporal match
    temporal_consistancy = ensemble_df.groupby("individual").apply(lambda x: x.ensembleTaxonID.value_counts().mean()).mean()
    comet_logger.experiment.log_metric("Temporal Consistancy",temporal_consistancy)
    
    correct = ensemble_df[ensemble_df.taxonID==ensemble_df.ensembleTaxonID]
    pos_temporal_consistancy = correct.groupby("individual").apply(lambda x: x.ensembleTaxonID.value_counts().mean()).mean()
    comet_logger.experiment.log_metric("True Positive Temporal Consistancy",pos_temporal_consistancy)

    incorrect = ensemble_df[~(ensemble_df.taxonID==ensemble_df.ensembleTaxonID)]
    neg_temporal_consistancy = incorrect.groupby("individual").apply(lambda x: x.ensembleTaxonID.value_counts().mean()).mean()
    comet_logger.experiment.log_metric("True Negative Temporal Consistancy",neg_temporal_consistancy)
    
if __name__ == "__main__":
    main()
