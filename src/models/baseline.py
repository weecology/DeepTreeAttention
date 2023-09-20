#Lightning Data Module
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
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
                
        return individual, y_hat 
           
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         factor=0.75,
                                                         verbose=True,
                                                         patience=8)
                                                                 
        return {'optimizer':optimizer, "lr_scheduler": {'scheduler': scheduler,"monitor":'val_loss',"frequency":self.config["validation_interval"],"interval":"epoch"}}
    
    def gather_predictions(self, predict_df):
        """Post-process the predict method to create metrics"""
        individuals = []
        yhats = []
        
        individuals = np.concatenate([batch[0] for batch in predict_df])
        predictions = np.concatenate([batch[1] for batch in predict_df])
        yhats = np.argmax(predictions, axis=1)
        scores = np.max(predictions, axis=1)

        results = pd.DataFrame({"individual":individuals,"yhat":yhats,"score":scores})
        
        return results
    
    def evaluation_scores(self, results, experiment):   
        results = results.groupby("individual").apply(lambda x: x.head(1))
        
        #Ensemble accuracy
        ensemble_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.yhat.values),
            target=torch.tensor(results.label.values),
            average="micro",
            task="multiclass",
            num_classes=self.classes
        )
            
        ensemble_macro_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.yhat.values),
            target=torch.tensor(results.label.values),
            average="macro",
            task="multiclass",
            num_classes=self.classes
        )
        
        ensemble_precision = torchmetrics.functional.precision(
            preds=torch.tensor(results.yhat.values),
            target=torch.tensor(results.label.values),
            num_classes=self.classes,
            task="multiclass"
        )
                        
        if experiment:
            experiment.log_metric("overall_macro", ensemble_macro_accuracy)
            experiment.log_metric("overall_micro", ensemble_accuracy) 
            experiment.log_metric("overal_precision", ensemble_precision) 
        
        #Species Accuracy
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.yhat.values),
            target=torch.tensor(results.label.values),
            average="none",
            num_classes=self.classes,
            task="multiclass"
        )
            
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(results.yhat.values),
            target=torch.tensor(results.label.values),
            average="none",
            num_classes=self.classes,
            task="multiclass"
        )        
        
        taxon_labels = list(self.label_to_index.keys())
        species_table = pd.DataFrame(
            {"taxonID":taxon_labels,
             "accuracy":taxon_accuracy,
             "precision":taxon_precision
             })
        
        if experiment:
            experiment.log_metrics(species_table.set_index("taxonID").accuracy.to_dict(),prefix="accuracy")
            experiment.log_metrics(species_table.set_index("taxonID").precision.to_dict(),prefix="precision")
                
        # Log result by site
        if experiment:
            site_data_frame =[]
            for name, group in results.groupby("siteID"):            
                site_micro = np.sum(group.yhat.values == group.label.values)/len(group.yhat.values)
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.yhat.values),
                    target=torch.tensor(group.label.values),
                    average="macro",
                    task="multiclass",
                    num_classes=self.classes)
                                
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro) 
                
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro], "Macro Recall": [site_macro]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)        
        
        return results