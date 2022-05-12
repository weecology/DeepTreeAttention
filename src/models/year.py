#Year model
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch
import torchmetrics
from src.models import Hang2020
from src.data import TreeDataset
import numpy as np
from pytorch_lightning import LightningModule
import pandas as pd

class base_model(Module):
    def __init__(self, classes, config):
        super().__init__()
        #Load from state dict of previous run
        if config["pretrain_state_dict"]:
            self.model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
        else:
            self.model = Hang2020.spectral_network(bands=config["bands"], classes=classes)
        
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes)
        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             })
        
    def forward(self,x):
        x = self.model(x)
        # Last attention layer as score        
        score = x[-1]
        
        return score 
    
class YearEnsemble(LightningModule):
    def __init__(self, classes, years, label_dict, config, loss_weight=None):
        super(YearEnsemble, self).__init__()        
        #Load from state dict of previous run
        self.years = years
        self.config = config
        self.models = nn.ModuleList()
        self.species_label_dict = label_dict
        self.classes = classes
        
        self.label_to_index = {}
        for x in label_dict:
            self.label_to_index[label_dict[x]] = x 
            
        for x in range(len(years)):
            m = base_model(classes=classes, config=config)
            self.models.append(m)
        
        if loss_weight is None:
            loss_weight = torch.ones(classes)
            
        if torch.cuda.is_available():
            self.loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
        else:
            self.loss_weight = torch.tensor(loss_weight, dtype=torch.float)
                    
        self.save_hyperparameters(ignore=["loss_weight"])    
        
    def configure_optimizers(self):
        """Create a optimizer for each level"""
        optimizers = []
        for x, year in enumerate(self.years):
            optimizer = torch.optim.Adam(self.models[x].parameters(), lr=self.config["lr".format(x)])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=0.75,
                                                             patience=5,
                                                             verbose=True,
                                                             threshold=0.0001,
                                                             threshold_mode='rel',
                                                             cooldown=0,
                                                             eps=1e-08)
            
            optimizers.append({'optimizer':optimizer, 'lr_scheduler': {"scheduler":scheduler, "monitor":'val_loss/dataloader_idx_{}'.format(x)}})

        return optimizers     
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Calculate train_df loss
        """
        individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[optimizer_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)    
        self.log("train_loss_{}".format(optimizer_idx),loss, on_epoch=True, on_step=False)

        return loss        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss 
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weight)   
        
        self.log("val_loss",loss)
        metric_dict = self.models[dataloader_idx].metrics(y_hat, y)
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        
        return {"individual":individual, "yhat":y_hat, "label":y}  
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Calculate predictions
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        y_hat = F.softmax(y_hat, dim=1)
        
        return {"individual":individual, "yhat":y_hat}  
        
    def validation_epoch_end(self, validation_step_outputs): 
        for index, results in enumerate(validation_step_outputs):
            year_yhat = torch.cat([x["yhat"] for x in results]).cpu()
            labels = torch.cat([x["label"] for x in results]).cpu()
            yhat = np.argmax(year_yhat, 1)
            
            epoch_micro = torchmetrics.functional.accuracy(
                preds=labels,
                target=yhat,
                average="micro")
            
            epoch_macro = torchmetrics.functional.accuracy(
                preds=labels,
                target=yhat,
                average="macro",
                num_classes=len(self.species_label_dict)
            )
            
            self.log("Epoch Micro Accuracy {}".format(self.years[index]), epoch_micro)
            self.log("Epoch Macro Accuracy {}".format(self.years[index]), epoch_macro)
        
    def ensemble(self, predict_ouputs):
        individual_dict ={}
        for index, results in enumerate(predict_ouputs):
            year_yhat = torch.cat([x["yhat"] for x in results]).cpu()
            individuals = np.concatenate([x["individual"] for x in results])
            for i, individual in enumerate(individuals):
                try:
                    individual_dict[individual].append(year_yhat[i])
                except:
                    individual_dict[individual] = [year_yhat[i]]
        pred = []
        scores = []
        for x in individual_dict:
            ensemble = torch.stack(individual_dict[x],axis=1).mean(axis=1).numpy()
            pred.append(np.argmax(ensemble))
            scores.append(np.max(ensemble))
        
        ensemble_df = pd.DataFrame({"individual":list(individual_dict.keys()),"pred_label_top1":pred,"top1_score":scores})
        ensemble_df["pred_taxa_top1"] = ensemble_df.pred_label_top1.apply(lambda x: self.label_to_index[x])
        
        return ensemble_df
        
    def ensemble_metrics(self, ensemble_df, experiment=None):
        # Log results by species
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.pred_label_top1.values),
            target=torch.tensor(ensemble_df.label.values), 
            average="none", 
            num_classes=self.classes
        )
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(ensemble_df.pred_label_top1.values),
            target=torch.tensor(ensemble_df.label.values),
            average="none",
            num_classes=self.classes
        )
        species_table = pd.DataFrame(
            {"taxonID":self.species_label_dict.keys(),
             "accuracy":taxon_accuracy,
             "precision":taxon_precision
             })
        
        if experiment:
            experiment.log_metrics(species_table.set_index("taxonID").accuracy.to_dict(),prefix="accuracy")
            experiment.log_metrics(species_table.set_index("taxonID").precision.to_dict(),prefix="precision")
                
        #Log result by site
        if experiment:
            site_data_frame =[]
            for name, group in ensemble_df.groupby("siteID"):                
                site_micro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.pred_label_top1.values),
                    target=torch.tensor(group.label.values),
                    average="micro")
                
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.pred_label_top1.values),
                    target=torch.tensor(group.label.values),
                    average="macro",
                    num_classes=self.classes)
                
                experiment.log_metric("{}_temporal_macro".format(name), site_macro)
                experiment.log_metric("{}_temporal_micro".format(name), site_micro) 
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro.numpy()], "Macro Recall": [site_macro.numpy()]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)        