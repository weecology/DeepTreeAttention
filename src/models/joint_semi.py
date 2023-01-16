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
from src import fixmatch
from src.models import baseline
from src import data, semi_supervised

class TreeModel(baseline.TreeModel):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self, model, classes, label_dict, supervised_train, supervised_test, loss_weight=None, config=None, client=None):
        super().__init__(model, classes, label_dict, loss_weight=None, config=None)
    
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
        
        # Unsupervised versus supervised loss weight
        self.alpha = torch.nn.Parameter(torch.tensor(self.config["semi_supervised"]["alpha"], dtype=float), requires_grad=False)
        if self.config["semi_supervised"]["semi_supervised_train"] is None:
            self.semi_supervised_train = semi_supervised.create_dataframe(config, label_to_taxon_id=self.index_to_label, client=client)
        else:
            self.semi_supervised_train = pd.read_csv(self.config["semi_supervised"]["semi_supervised_train"])
            
        self.supervised_train = supervised_train
        self.supervised_test = supervised_test
        
        #Create model 
        self.model = model
        
        #Metrics
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes)
        top_k_recall = torchmetrics.Accuracy(average="micro",top_k=self.config["top_k"])

        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             "Top {} Accuracy".format(self.config["top_k"]): top_k_recall
             })

        self.save_hyperparameters(ignore=["loss_weight","client"])
        
        #Weighted loss - on reload and loss_weight = None, this is skipped
        if loss_weight is None:
            loss_weight = torch.ones((classes))   
        try:
            if torch.cuda.is_available():
                self.loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
            else:
                self.loss_weight = torch.ones((classes))    
        except:
            pass
        
    def train_dataloader(self):
        ## Labeled data
        
        labeled_ds = data.TreeDataset(
            df=self.supervised_train,
            config=self.config,
            train=True
        )
        
        self.data_loader = torch.utils.data.DataLoader(
            labeled_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        ## Unlabeled data
        semi_supervised_config = copy.deepcopy(self.config)
        semi_supervised_config["crop_dir"] = semi_supervised_config["semi_supervised"]["crop_dir"]

        unlabeled_ds = fixmatch.TreeDataset(
            df=self.semi_supervised_train,
            config=semi_supervised_config
        )
        
        self.unlabeled_data_loader = torch.utils.data.DataLoader(
            unlabeled_ds,
            batch_size=self.config["semi_supervised"]["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return {"labeled":self.data_loader, "unlabeled": self.unlabeled_data_loader}
        
    
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Labeled data
        individual, inputs, y = batch["labeled"]
        labeled_images = inputs["HSI"]
        y_hat = self.model.forward(labeled_images)
        supervised_loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)   
        
        ## Unlabeled data - Weak Augmentation
        individual, inputs = batch["unlabeled"]
        unlabeled_images = inputs["Weak"]
        
        #Combine labeled and unlabeled data to preserve batchnorm
        logit_weak = self.model.forward(unlabeled_images)  
        prob_weak = F.softmax(logit_weak, dim=1)
        
        # Unlabeled data - Strong Augmentation
        images = inputs["Strong"]
        logit_strong = self.model.forward(images)
        
        #Only select those labels greater than threshold
        p_pseudo_label, pseudo_label = torch.max(prob_weak.detach(), dim=-1)
        threshold_mask = p_pseudo_label.ge(self.config["semi_supervised"]["fixmatch_threshold"]).float()
        pseudo_loss = F.cross_entropy(logit_strong, pseudo_label, reduction="none")
        pseudo_loss = (pseudo_loss * threshold_mask).mean()
        self.unlabeled_samples_count = self.unlabeled_samples_count + sum(threshold_mask)
        
        self.log("Unlabeled mean training confidence",p_pseudo_label.mean())            
        self.log("supervised_loss",supervised_loss, on_step=True)
        self.log("unsupervised_loss", pseudo_loss, on_step=True)
        
        if self.current_epoch > 100:
            loss = supervised_loss + (self.alpha * pseudo_loss) 
        else:
            loss = supervised_loss
        
        return loss
    
    def on_train_epoch_start(self):
        """Reset count of unlabeled samples per train epoch"""
        self.unlabeled_samples_count = 0
    
    def on_train_epoch_end(self):
        self.log("unlabeled_samples",self.unlabeled_samples_count)
        
    def val_dataloader(self):
        """Validation data loader only includes labeled data"""
        
        val_ds = data.TreeDataset(
            df = self.supervised_test,
            config=self.config
        )
        
        data_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
        )
        
        return data_loader
    
    def predict_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
        predicted_class = F.softmax(y_hat, dim=1)
        
        return predicted_class
