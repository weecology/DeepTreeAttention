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
from src import utils

class TreeModel(LightningModule):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self, model, classes, label_dict, loss_weight=None, config=None, *args, **kwargs):
        super().__init__()
    
        self.ROOT = os.path.dirname(os.path.dirname(__file__))    
        if config is None:
            self.config = utils.read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config
        
        self.classes = classes
        self.label_to_index = label_dict
        self.index_to_label = {}
        for x in label_dict:
            self.index_to_label[label_dict[x]] = x 
        
        # Create model 
        self.model = model
        
        # Metrics
        micro_recall = torchmetrics.Accuracy(average="micro", num_classes=classes, task="multiclass")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes,  task="multiclass")
        top_k_recall = torchmetrics.Accuracy(average="micro",top_k=self.config["top_k"],   num_classes=classes, task="multiclass")

        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             "Top {} Accuracy".format(self.config["top_k"]): top_k_recall
             })

        self.save_hyperparameters(ignore=["loss_weight"])
        
        #Weighted loss
        if torch.cuda.is_available():
            self.loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
        else:
            self.loss_weight = torch.tensor(loss_weight, dtype=torch.float)
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)    

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        
        return loss

    def predict_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
                
        return y_hat 
           
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.75,
                                                         verbose=True,
                                                         patience=8)
                                                                 
        return {'optimizer':optimizer, "lr_scheduler": {'scheduler': scheduler,"monitor":'val_loss',"frequency":self.config["validation_interval"],"interval":"epoch"}}
            