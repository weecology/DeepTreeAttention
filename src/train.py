# Train
import comet_ml
import os
import numpy as np
from pandas.util import hash_pandas_object
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning.callbacks import LearningRateMonitor

from src import data, start_cluster, metrics
from src.models import baseline, Hang2020

def main(git_branch, git_commit, config, site=None):
    #Create datamodule
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
    if site:
        tag = "{}_{}".format(git_branch, site)
    else:
        tag = git_branch
    comet_logger.experiment.add_tag(tag)
    comet_logger.experiment.log_parameter("commit hash",git_commit)
    comet_logger.experiment.log_parameters(config)
    
    data_module = data.TreeData(
        csv_file="../data/raw/neon_vst_data_2022.csv",
        data_dir=config["crop_dir"],
        config=config,
        client=client,
        site=site,
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
    
    train_model(data_module, comet_logger, site)
    
def train_model(data_module, comet_logger, name):
    with comet_logger.experiment.context_manager(name):
        print(data_module.species_label_dict)
        loss_weight = []
        for x in data_module.species_label_dict:
            try:
                lw = 1/data_module.train[data_module.train.taxonID==x].shape[0]
            except:
                lw = 0
            loss_weight.append(lw)
        loss_weight = np.array(loss_weight/np.max(loss_weight))
        loss_weight[loss_weight < 0.5] = 0.5  
        
        comet_logger.experiment.log_parameter("loss_weight", loss_weight)
        
        # Create Model
        model = Hang2020.Single_Spectral_Model(bands=data_module.config["bands"], classes=data_module.num_classes)      
        
        if data_module.config["pretrain_state_dict"]:
            model.state_dict(torch.load(data_module.config["pretrain_state_dict"]))
            
        m = baseline.TreeModel(
            model=model, 
            config=data_module.config,
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
            check_val_every_n_epoch=data_module.config["validation_interval"],
            callbacks=[lr_monitor],
            enable_checkpointing=False,
            logger=comet_logger)
        
        trainer.fit(m, datamodule=data_module)
        
        #Save model checkpoint
        if data_module.config["snapshot_dir"] is not None:
            trainer.save_checkpoint("{}_{}.pt".format(data_module.config["snapshot_dir"], git_commit, name))
            torch.save(m.model.state_dict(), "{}/{}_{}_state_dict.pt".format(data_module.config["snapshot_dir"], git_commit, name))
        
        results = m.evaluate_crowns(
            data_loader=data_module.val_dataloader(),
            siteIDs=data_module.test.siteID,
            experiment=comet_logger.experiment,
        )
        
        comet_logger.experiment.log_confusion_matrix(
            results.label.values,
            results.pred_label_top1.values,
            labels=list(data_module.species_label_dict.keys()),
            max_categories=len(data_module.species_label_dict.keys()),
            title=name,
            file_name="{}.json".format(name)
        )
        
        # Log prediction
        comet_logger.experiment.log_table("test_predictions.csv", results)

        # Within plot confusion
        plot_lists = data_module.train.groupby("label").plotID.unique()
        within_plot_confusion = metrics.site_confusion(y_true = results.label, y_pred = results.pred_label_top1, site_lists=plot_lists)
        comet_logger.experiment.log_metric("within_plot_confusion", within_plot_confusion)
