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
    def __init__(self, model, classes, label_dict, supervised_train, supervised_test, loss_weight=None, config=None):
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
            self.semi_supervised_train = semi_supervised.create_dataframe(config, label_to_taxon_id=self.index_to_label)
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

        self.save_hyperparameters(ignore=["loss_weight"])
        
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
        semi_supervised_config = copy.deepcopy(self.config)
        semi_supervised_config["crop_dir"] = semi_supervised_config["semi_supervised"]["crop_dir"]
        semi_supervised_config["preload_images"] = semi_supervised_config["semi_supervised"]["preload_images"]
        semi_supervised_config["workers"] = semi_supervised_config["semi_supervised"]["workers"]

        unlabeled_ds = fixmatch.TreeDataset(
            df=self.semi_supervised_train,
            config=semi_supervised_config,
            train=True
        )
        
        labeled_ds = data.TreeDataset(
            df=self.supervised_train,
            config=self.config,
            train=True
        )
        
        #labeled 
        data_loader = torch.utils.data.DataLoader(
            labeled_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        #Unlabeled
        unlabeled_data_loader = torch.utils.data.DataLoader(
            unlabeled_ds,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=self.config["workers"],
        )
        
        return {"labeled":data_loader, "unlabeled": unlabeled_data_loader}
    
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Labeled data
        individual, inputs, y = batch["labeled"]
        images = inputs["HSI"]
        y_hat = self.model.forward(images)
        supervised_loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)   
        
        # Unlabeled data - Weak Augmentation
        individual, inputs, y = batch["unlabeled"]
        images = inputs["Weak"]
        y_hat_weak = self.model.forward(images)    
        
        # Unlabeled data - Strong Augmentation
        individual, inputs, y = batch["unlabeled"]
        images = inputs["Strong"]
        y_hat_strong = self.model.forward(images)
        
        #Only select those labels greater than threshold
        samples_to_keep = torch.max(y_hat_strong, dim=1).values > self.config["semi_supervised"]["fixmatch_threshold"]
        selected_unlabeled_yhat = y_hat_strong[samples_to_keep,:]
        
        if selected_unlabeled_yhat.shape[0] > 0:
            selected_weak_y = y_hat_weak[samples_to_keep,:]
            psuedo_label = torch.argmax(selected_weak_y, dim=1)
            
            #Useful to log number of samples kept
            self.log("unlabeled_samples",selected_weak_y.shape[0])
            unsupervised_loss = F.cross_entropy(input=selected_unlabeled_yhat, target=psuedo_label)    
        else:
            unsupervised_loss = 0
            
        self.log("supervised_loss",supervised_loss)
        self.log("unsupervised_loss", unsupervised_loss)
        self.log("alpha", self.alpha, on_step=False, on_epoch=True)
        loss = supervised_loss + self.alpha * unsupervised_loss 
        
        return loss
    
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
    
    def validation_step(self, batch, batch_idx):
        """Val on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)        
        self.log_dict(self.metrics(y_hat, y), on_epoch=True, on_step=False)
        
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
        predicted_class = F.softmax(y_hat, dim=1)
        
        return predicted_class    
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.75,
                                                         patience=5,
                                                         min_lr=0.000001,
                                                         verbose=True)
                                                                 
        return {'optimizer':optimizer, 'lr_scheduler': {"scheduler":scheduler,"monitor":'val_loss',"frequency":self.config["validation_interval"], "interval": "epoch"}}
    

    def predict(self,inputs):
        """Given a input dictionary, construct args for prediction"""
        if "cuda" == self.device.type:
            images = inputs["HSI"]
            images = [x.cuda() for x in images]
            pred = self.model(images)
            pred = pred.cpu()
        else:
            images = inputs["HSI"]
            pred = self.model(images)
        
        return pred