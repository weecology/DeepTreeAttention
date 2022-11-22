#Multiple stage model
from src.models.year import learned_ensemble
from src.data import TreeDataset
from src import utils

from pytorch_lightning import LightningModule
import pandas as pd
import math
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torchmetrics
from torchmetrics import Accuracy, ClasswiseWrapper, Precision, MetricCollection
import torch

class base_model(Module):
    def __init__(self, years, classes, config):
        super().__init__()
        #Load from state dict of previous run
        self.model = learned_ensemble(classes=classes, years=years, config=config)
        micro_recall = Accuracy(average="micro")
        macro_recall = Accuracy(average="macro", num_classes=classes)
        self.metrics = MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             })
        
    def forward(self,x):
        score = self.model(x)        
        
        return score 
    
class MultiStage(LightningModule):
    def __init__(self, train_df, test_df, crowns, config, train_mode=True, debug=False):
        super().__init__()
        # Generate each model
        self.years = train_df.tile_year.unique()
        self.config = config
        self.models = nn.ModuleList()
        self.species_label_dict = train_df[["taxonID","label"]].drop_duplicates().set_index("taxonID").to_dict()["label"]
        self.index_to_label = {v:k for k,v in self.species_label_dict.items()}
        self.crowns = crowns
        self.level_label_dicts = []    
        self.label_to_taxonIDs = []   
        self.train_df = train_df
        self.test_df = test_df
                
        #hotfix for old naming schema
        try:
            self.test_df["individual"] = self.test_df["individualID"]
            self.train_df["individual"] = self.train_df["individualID"]
        except:
            pass
        
        if train_mode:
            self.train_datasets, self.test_datasets = self.create_datasets()
            self.levels = len(self.train_datasets)       
            
            #Generate metrics for each class level
            self.level_metrics = nn.ModuleDict()
            for level in self.level_id:
                taxon_level_labels = list(self.level_label_dicts[level].keys())
                num_classes = len(self.level_label_dicts[level])
                level_metric = MetricCollection({       
                "Species accuracy":ClasswiseWrapper(Accuracy(average="none", num_classes=num_classes), labels=taxon_level_labels),
                "Species precision":ClasswiseWrapper(Precision(average="none", num_classes=num_classes),labels=taxon_level_labels),
                })
                self.level_metrics["level_{}".format(level)] = level_metric

            self.classes = len(self.train_df.label.unique())
            for index, ds in enumerate([self.level_0_train, self.level_1_train]): 
                labels = ds.label
                classes = self.num_classes[index]
                base = base_model(classes=classes, years=len(self.years), config=self.config)
                self.models.append(base)            
                loss_weight = []
                for x in range(classes):
                    try:
                        w = 1/np.sum(labels==x)
                    except:
                        w = 1 
                    loss_weight.append(w)
        
                loss_weight = np.array(loss_weight/np.max(loss_weight))
                loss_weight[loss_weight < self.config["min_loss_weight"]] = self.config["min_loss_weight"] 
                loss_weight = torch.tensor(loss_weight, dtype=torch.float)                        
                pname = 'loss_weight_{}'.format(index)            
                self.register_buffer(pname, loss_weight)
            if not debug:
                self.save_hyperparameters()        
            
    def create_datasets(self):
        #Create levels for each year
        ## Level 0     
        train_datasets = []
        test_datasets = []
        self.num_classes = []
        self.level_id = []
   
        # Level 0, the most common species at each site
        self.level_0_train = self.train_df.copy()
        common_species = self.level_0_train.taxonID.value_counts().reset_index()
        common_species = common_species[common_species.taxonID > self.config["head_class_minimum_samples"]]["index"]
        self.level_label_dicts.append({value:key for key, value in enumerate(common_species)})
        self.level_label_dicts[0]["OTHER"] = len(self.level_label_dicts[0])
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[0].items()})
        
        # Select head and tail classes
        head_classes = self.level_0_train[self.level_0_train.taxonID.isin(common_species)]
        tail_classes = self.level_0_train[~self.level_0_train.taxonID.isin(common_species)]
        tail_classes["taxonID"] = "OTHER"
        
        # Create labels
        self.level_0_train = pd.concat([head_classes, tail_classes])                
        self.level_0_train["label"] = [self.level_label_dicts[0][x] for x in self.level_0_train.taxonID]
        self.level_0_train_ds = TreeDataset(df=self.level_0_train, config=self.config)
        train_datasets.append(self.level_0_train_ds)
        self.num_classes.append(len(self.level_0_train.taxonID.unique()))
        
        self.level_0_test = self.test_df.copy()
        head_classes = self.level_0_test[self.level_0_test.taxonID.isin(common_species)]
        tail_classes = self.level_0_test[~self.level_0_test.taxonID.isin(common_species)]
        tail_classes["taxonID"] = "OTHER"
        self.level_0_test = pd.concat([head_classes, tail_classes]) 
        
        self.level_0_test["label"]= [self.level_label_dicts[0][x] for x in self.level_0_test.taxonID]            
        self.level_0_test_ds = TreeDataset(df=self.level_0_test, config=self.config, train=True)
        test_datasets.append(self.level_0_test_ds)
        self.level_id.append(0)

        # Level 1, the remaining species
        self.level_1_train = self.train_df.copy()
        rare_species = [x for x in self.train_df.taxonID.unique() if x not in common_species.values]
        self.level_label_dicts.append({value:key for key, value in enumerate(rare_species)})
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[1].items()})
        
        # Select head and tail classes
        tail_classes = self.level_1_train[self.level_1_train.taxonID.isin(rare_species)]
        
        # Create labels
        self.level_1_train = tail_classes.groupby("taxonID").apply(lambda x: x.head(self.config["rare_class_sampling_max"]))              
        self.level_1_train["label"] = [self.level_label_dicts[1][x] for x in self.level_1_train.taxonID]
        self.level_1_train_ds = TreeDataset(df=self.level_1_train, config=self.config)
        train_datasets.append(self.level_1_train_ds)
        self.num_classes.append(len(self.level_1_train.taxonID.unique()))
        
        self.level_1_test = self.test_df.copy()
        tail_classes = self.level_1_test[self.level_1_test.taxonID.isin(rare_species)]
        self.level_1_test = tail_classes
        
        self.level_1_test["label"]= [self.level_label_dicts[1][x] for x in self.level_1_test.taxonID]            
        self.level_1_test_ds = TreeDataset(df=self.level_1_test, config=self.config)
        test_datasets.append(self.level_1_test_ds)
        self.level_id.append(1)
        
        return train_datasets, test_datasets
    
    def train_dataloader(self):
        data_loaders = []
        for ds in self.train_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"]
            )
            data_loaders.append(data_loader)
        
        return data_loaders        

    def val_dataloader(self):
        data_loaders = []
        for ds in self.test_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["workers"]
            )
            data_loaders.append(data_loader)
        
        return data_loaders 
    
    def predict_dataloader(self, ds):
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["predict_batch_size"],
            shuffle=False,
            num_workers=self.config["workers"]
        )

        return data_loader
        
    def configure_optimizers(self):
        """Create a optimizer for each level"""
        optimizers = []
        for x, ds in enumerate(self.train_datasets):
            optimizer = torch.optim.Adam(self.models[x].parameters(), lr=self.config["lr_{}".format(x)])
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                             mode='min',
                                                             factor=0.75,
                                                             patience=8,
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
        #get loss weight
        loss_weights = self.__getattr__('loss_weight_'+str(optimizer_idx))
        
        individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[optimizer_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=loss_weights)    
        self.log("train_loss_{}".format(optimizer_idx),loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss and on_epoch metrics
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y)   
        
        self.log("val_loss",loss)
        self.models[dataloader_idx].metrics(y_hat, y)
        self.log_dict(self.models[dataloader_idx].metrics, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
   
        self.level_metrics["level_{}".format(dataloader_idx)].update(y_hat, y)
 
        return {"individual":individual, "yhat":y_hat, "label":y}  
    
    def predict_step(self, batch, batch_idx):
        """Calculate predictions
        """
        individual, inputs = batch
        images = inputs["HSI"]  
        
        y_hats = []
        for model in self.models:   
            y_hat = model.forward(images)
            y_hat = F.softmax(y_hat, dim=1)
            y_hats.append(y_hat)
        
        return individual, y_hats
    
    def on_predict_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)
    
    def on_validation_epoch_end(self):
        for level in self.level_id:
            class_metrics = self.level_metrics["level_{}".format(level)].compute()
            self.log_dict(class_metrics, on_epoch=True, on_step=False)
            self.level_metrics["level_{}".format(level)].reset()
        
    def gather_predictions(self, predict_df):
        """Post-process the predict method to create metrics"""
        individuals = []
        yhats = []
        levels = []
        
        for output in predict_df:
            for index, level_results in enumerate(output[1]):
                batch_individuals = np.stack(output[0])
                for individual, yhat in zip(batch_individuals, level_results):
                    individuals.append(individual)                
                    yhats.append(yhat)
                    levels.append(index)
                
        temporal_average = pd.DataFrame({"individual":individuals,"level":levels,"yhat":yhats})
                
        #Argmax and score for each level
        predicted_label = temporal_average.groupby(["individual","level"]).yhat.apply(
            lambda x: np.argmax(np.vstack(x))).reset_index().pivot(
                index=["individual"],columns="level",values="yhat").reset_index()
        predicted_label.columns = ["individual","pred_label_top1_level_0","pred_label_top1_level_1"]
        
        predicted_score = temporal_average.groupby(["individual","level"]).yhat.apply(
            lambda x: np.vstack(x).max()).reset_index().pivot(
                index=["individual"],columns="level",values="yhat").reset_index()
        predicted_score.columns = ["individual","top1_score_level_0","top1_score_level_1"]
        results = pd.merge(predicted_label,predicted_score)
        
        #Label taxa
        for level, label_dict in enumerate(self.label_to_taxonIDs):
            results["pred_taxa_top1_level_{}".format(level)] = results["pred_label_top1_level_{}".format(level)].apply(lambda x: label_dict[x])
        
        return results
    
    def ensemble(self, results):
        """Given a multi-level model, create a final output prediction and score"""
        ensemble_taxonID = []
        ensemble_label = []
        ensemble_score = []
        
        #For each level, select the predicted taxonID and retrieve the original label order
        for index,row in results.iterrows():
            if not row["pred_taxa_top1_level_0"] == "OTHER":
                ensemble_taxonID.append(row["pred_taxa_top1_level_0"])
                ensemble_label.append(self.species_label_dict[row["pred_taxa_top1_level_0"]])
                ensemble_score.append(row["top1_score_level_0"])                
            else:
                ensemble_taxonID.append(row["pred_taxa_top1_level_1"])
                ensemble_label.append(self.species_label_dict[row["pred_taxa_top1_level_1"]])
                ensemble_score.append(row["top1_score_level_1"])                   

        results["ensembleTaxonID"] = ensemble_taxonID
        results["ens_score"] = ensemble_score
        results["ens_label"] = ensemble_label   
        
        return results
            
    def evaluation_scores(self, ensemble_df, experiment):   
        ensemble_df = ensemble_df.groupby("individual").apply(lambda x: x.head(1))
        
        #Ensemble accuracy
        ensemble_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="micro",
            num_classes=len(self.species_label_dict)
        )
            
        ensemble_macro_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="macro",
            num_classes=len(self.species_label_dict)
        )
        
        ensemble_precision = torchmetrics.functional.precision(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            num_classes=len(self.species_label_dict)
        )
                        
        if experiment:
            experiment.log_metric("ensemble_macro", ensemble_macro_accuracy)
            experiment.log_metric("ensemble_micro", ensemble_accuracy) 
            experiment.log_metric("ensemble_precision", ensemble_precision) 
        
        #Species Accuracy
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="none",
            num_classes=len(self.species_label_dict)
        )
            
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="none",
            num_classes=len(self.species_label_dict)
        )        
        
        taxon_labels = list(self.species_label_dict)
        taxon_labels.sort()
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
            for name, group in ensemble_df.groupby("siteID"):            
                site_micro = np.sum(group.ens_label.values == group.label.values)/len(group.ens_label.values)
                
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.ens_label.values),
                    target=torch.tensor(group.label.values),
                    average="macro",
                    num_classes=len(self.species_label_dict))
                                
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro) 
                
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro], "Macro Recall": [site_macro]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)        
        
        return ensemble_df