#Lightning Data Module
from . import __file__
import numpy as np
from pytorch_lightning import LightningModule
import os
import pandas as pd
from torch.nn import functional as F
from torch import optim
import torch
import torchmetrics
import copy

from src import utils
from src.models import multi_stage
from src import data, semi_supervised

class TreeModel(multi_stage.MultiStage):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self, train_df, test_df, taxonomic_csv, config=None):
        super().__init__(train_df=train_df, test_df=test_df, taxonomic_csv=taxonomic_csv, config=config)
    
        # Unsupervised versus supervised loss weight
        self.alpha = torch.nn.Parameter(torch.tensor(self.config["semi_supervised"]["alpha"], dtype=float), requires_grad=False)
        if self.config["semi_supervised"]["semi_supervised_train"] is None:
            self.semi_supervised_train = semi_supervised.create_dataframe(config, label_to_taxon_id=self.index_to_label)
        else:
            self.semi_supervised_train = pd.read_csv(self.config["semi_supervised"]["semi_supervised_train"])
        
        print("Shape of semi-supervised {}".format(self.semi_supervised_train.shape))
        
        # Hierarchical psuedolabels
        self.psuedo_dataframes, _ = self.create_levels(self.semi_supervised_train, level_label_dicts=self.level_label_dicts)
        
    def train_dataloader(self):
        semi_supervised_config = copy.deepcopy(self.config)
        semi_supervised_config["crop_dir"] = semi_supervised_config["semi_supervised"]["crop_dir"]
        semi_supervised_config["preload_images"] = semi_supervised_config["semi_supervised"]["preload_images"]
        semi_supervised_config["workers"] = semi_supervised_config["semi_supervised"]["workers"]

        train_dataloaders = []
        for level, ds in enumerate(self.train_datasets):
            # labeled 
            labeled_data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            
            # Unlabeled
            pseudo_dataset = data.TreeDataset(df=self.psuedo_dataframes[level], config=semi_supervised_config)
            unlabeled_data_loader = torch.utils.data.DataLoader(
                pseudo_dataset,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            train_dataloaders.append({"labeled":labeled_data_loader, "unlabeled": unlabeled_data_loader})
        
        return train_dataloaders
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Train on a loaded dataset
        """
        
        # Labeled data
        loss_weights = self.__getattr__('loss_weight_'+str(optimizer_idx))        
        individual, inputs, y = batch[optimizer_idx]["labeled"]
        images = inputs["HSI"]
        y_hat = self.models[optimizer_idx].forward(images)
        supervised_loss = F.cross_entropy(y_hat, y, weight=loss_weights)   
        
        # Unlabeled data
        individual, inputs, y = batch[optimizer_idx]["unlabeled"]
        images = inputs["HSI"]
        y_hat = self.models[optimizer_idx].forward(images)        
        unsupervised_loss = F.cross_entropy(y_hat, y)    
        
        self.log("supervised_loss_level_{}".format(optimizer_idx),supervised_loss)
        self.log("unsupervised_loss_level_{}".format(optimizer_idx), unsupervised_loss)
        self.log("alpha", self.alpha, on_step=False, on_epoch=True)
        loss = supervised_loss + self.alpha * unsupervised_loss 
        
        return loss