#Multiple stage model
from functools import reduce
import geopandas as gpd
from src.models import Hang2020
from src.data import TreeDataset
from pytorch_lightning import LightningModule
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torchmetrics
import torch

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
    
class MultiStage(LightningModule):
    def __init__(self, train_df, test_df, crowns, config):
        super().__init__()        
        # Generate each model
        self.years = train_df.tile_year.unique()
        self.levels = 5        
        self.loss_weights = []
        self.config = config
        self.models = nn.ModuleList()
        self.species_label_dict = train_df[["taxonID","label"]].drop_duplicates().set_index("taxonID").to_dict()["label"]
        self.index_to_label = {v:k for k,v in self.species_label_dict.items()}
        self.crowns = crowns
        self.level_label_dicts = {}     
        self.label_to_taxonIDs = {}    
        self.train_df = train_df
        self.test_df = test_df
        self.train_datasets, self.test_datasets = self.create_datasets()
        self.classes = len(self.train_df.label.unique())
        
        for ds in self.train_datasets: 
            labels = [x[3] for x in ds]
            base = base_model(classes=self.classes, config=config)
            loss_weight = []
            for x in np.unique(labels):
                loss_weight.append(1/np.sum(labels==x))

            loss_weight = np.array(loss_weight/np.max(loss_weight))
            loss_weight[loss_weight < 0.5] = 0.5  
            
            if torch.cuda.is_available():
                loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
            else:
                loss_weight = torch.tensor(loss_weight, dtype=torch.float)
                
            self.loss_weights.append(torch.tensor(loss_weight))
            self.models.append(base)
            
        self.save_hyperparameters(ignore=["loss_weight"])
        
    def create_datasets(self):
        #Create levels for each year
        ## Level 0     
        train_datasets = []
        test_datasets = []
        self.level_test_dict = {}
        for level in range(self.levels):
            self.level_test_dict[level] = []
        
        for year in self.years:
            self.level_label_dicts[0] = {"PIPA2":0,"OTHER":1}
            self.label_to_taxonIDs[0] = {v: k  for k, v in self.level_label_dicts[0].items()}
            
            self.level_0_train = self.train_df.copy()
            self.level_0_train.loc[~(self.level_0_train.taxonID == "PIPA2"),"taxonID"] = "OTHER"
            self.level_0_train.loc[(self.level_0_train.taxonID == "PIPA2"),"taxonID"] = "PIPA2"  
            self.level_0_train = self.level_0_train.groupby("taxonID").apply(lambda x:x.sample(frac=1).head(self.config["PIPA2_sampling_ceiling"])).reset_index(drop=True)        
            self.level_0_train["label"] = [self.level_label_dicts[0][x] for x in self.level_0_train.taxonID]
            self.level_0_train_ds = TreeDataset(df=self.level_0_train, config=self.config, year=year)
            train_datasets.append(self.level_0_train_ds)
            
            self.level_0_test = self.test_df.copy()
            self.level_0_test.loc[~(self.level_0_test.taxonID == "PIPA2"),"taxonID"] = "OTHER"
            self.level_0_test.loc[(self.level_0_test.taxonID == "PIPA2"),"taxonID"] = "PIPA2"                        
            self.level_0_test["label"]= [self.level_label_dicts[0][x] for x in self.level_0_test.taxonID]            
            self.level_0_test_ds = TreeDataset(df=self.level_0_test, config=self.config, year=year)
            test_datasets.append(self.level_0_test_ds)
            self.level_test_dict[0].append(self.level_0_test_ds)
            
            ## Level 1
            self.level_label_dicts[1] =  {"CONIFER":0,"BROADLEAF":1}
            self.label_to_taxonIDs[1] = {v: k  for k, v in self.level_label_dicts[1].items()}
            self.level_1_train = self.train_df.copy()
            self.level_1_train = self.level_1_train[~(self.level_1_train.taxonID=="PIPA2")]    
            self.level_1_train.loc[~self.level_1_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"   
            self.level_1_train.loc[self.level_1_train.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER" 
            
            #subsample broadleaf, labels have not been converted, relate to original taxonID
            broadleaf_ids = self.level_1_train[self.level_1_train.taxonID=="BROADLEAF"].groupby("label").apply(lambda x: x.sample(frac=1).head(self.config["broadleaf_ceiling"])).individualID
            conifer_ids = self.level_1_train[self.level_1_train.taxonID=="CONIFER"].individualID
            ids_to_keep = np.concatenate([broadleaf_ids, conifer_ids])
            self.level_1_train = self.level_1_train[self.level_1_train.individualID.isin(ids_to_keep)].reset_index(drop=True)
            self.level_1_train["label"] = [self.level_label_dicts[1][x] for x in self.level_1_train.taxonID]
            self.level_1_train_ds = TreeDataset(df=self.level_1_train, config=self.config, year=year)
            train_datasets.append(self.level_1_train_ds)
            
            self.level_1_test = self.test_df.copy()
            self.level_1_test = self.level_1_test[~(self.level_1_test.taxonID=="PIPA2")].reset_index(drop=True)    
            self.level_1_test.loc[~self.level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"   
            self.level_1_test.loc[self.level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"            
            self.level_1_test["label"] = [self.level_label_dicts[1][x] for x in self.level_1_test.taxonID]
            self.level_1_test_ds = TreeDataset(df=self.level_1_test, config=self.config, year=year)
            test_datasets.append(self.level_1_test_ds)
            self.level_test_dict[1].append(self.level_1_test_ds)
            
            ## Level 2
            broadleaf = [x for x in list(self.species_label_dict.keys()) if (not x in ["PICL","PIEL","PITA","PIPA2"]) & (not "QU" in x)]            
            self.level_label_dicts[2] =  {v:k for k, v in enumerate(broadleaf)}
            self.level_label_dicts[2]["OAK"] = len(self.level_label_dicts[2])
            self.label_to_taxonIDs[2] = {v: k  for k, v in self.level_label_dicts[2].items()}
                        
            self.level_2_train = self.train_df.copy()
            self.level_2_train = self.level_2_train[~self.level_2_train.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])].reset_index(drop=True)
            self.level_2_train.loc[self.level_2_train.taxonID.str.contains("QU"),"taxonID"] = "OAK"
            self.level_2_train["label"] = [self.level_label_dicts[2][x] for x in self.level_2_train.taxonID]
            self.level_2_train_ds = TreeDataset(df=self.level_2_train, config=self.config, year=year)
            train_datasets.append(self.level_2_train_ds)
            
            self.level_2_test = self.test_df.copy()
            self.level_2_test = self.level_2_test[~self.level_2_test.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])].reset_index(drop=True) 
            self.level_2_test.loc[self.level_2_test.taxonID.str.contains("QU"),"taxonID"] = "OAK"
            self.level_2_test["label"] = [self.level_label_dicts[2][x] for x in self.level_2_test.taxonID]
            self.level_2_test_ds = TreeDataset(df=self.level_2_test, config=self.config, year=year)
            test_datasets.append(self.level_2_test_ds)
            self.level_test_dict[2].append(self.level_2_test_ds)
            
            ## Level 3
            evergreen = [x for x in list(self.species_label_dict.keys()) if x in ["PICL","PIEL","PITA"]]         
            self.level_label_dicts[3] =  {v:k for k, v in enumerate(evergreen)}
            self.label_to_taxonIDs[3] = {v: k  for k, v in self.level_label_dicts[3].items()}
                        
            self.level_3_train = self.train_df.copy()
            self.level_3_train = self.level_3_train[self.level_3_train.taxonID.isin(["PICL","PIEL","PITA"])].reset_index(drop=True) 
            self.level_3_train["label"] = [self.level_label_dicts[3][x] for x in self.level_3_train.taxonID]
            self.level_3_train_ds = TreeDataset(df=self.level_3_train, config=self.config)
            train_datasets.append(self.level_3_train_ds)
            
            self.level_3_test = self.test_df.copy()
            self.level_3_test = self.level_3_test[self.level_3_test.taxonID.isin(["PICL","PIEL","PITA"])].reset_index(drop=True) 
            self.level_3_test["label"] = [self.level_label_dicts[3][x] for x in self.level_3_test.taxonID]
            self.level_3_test_ds = TreeDataset(df=self.level_3_test, config=self.config, year=year)
            test_datasets.append(self.level_3_test_ds)
            self.level_test_dict[3].append(self.level_3_test_ds)
            
            ## Level 4
            oak = [x for x in list(self.species_label_dict.keys()) if "QU" in x]
            self.level_label_dicts[4] =  {v:k for k, v in enumerate(oak)}
            self.label_to_taxonIDs[4] = {v: k  for k, v in self.level_label_dicts[4].items()}
            
            #Balance the train in OAKs
            self.level_4_train = self.train_df.copy()
            self.level_4_train = self.level_4_train[self.level_4_train.taxonID.str.contains("QU")].reset_index(drop=True)
            self.level_4_train["label"] = [self.level_label_dicts[4][x] for x in self.level_4_train.taxonID]
            self.level_4_train = self.level_4_train.groupby("taxonID").apply(lambda x:x.sample(frac=1).head(self.config["oaks_sampling_ceiling"])).reset_index(drop=True)
            self.level_4_train_ds = TreeDataset(df=self.level_4_train, config=self.config, year=year)
            train_datasets.append(self.level_4_train_ds)

            self.level_4_test = self.test_df.copy()
            self.level_4_test = self.level_4_test[self.level_4_test.taxonID.str.contains("QU")].reset_index(drop=True)
            self.level_4_test["label"] = [self.level_label_dicts[4][x] for x in self.level_4_test.taxonID]
            self.level_4_test_ds = TreeDataset(df=self.level_4_test, config=self.config, year=year)
            test_datasets.append(self.level_4_test_ds)
            self.level_test_dict[4].append(self.level_4_test_ds)

        return train_datasets, test_datasets
    
    def train_dataloader(self):
        data_loaders = {}
        for ds in self.train_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            data_loaders[ds] = data_loader
        
        return data_loaders        

    def val_dataloader(self):
        ## Validation loaders are a list https://github.com/PyTorchLightning/pytorch-lightning/issues/10809
        data_loaders = []
        for ds in self.test_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["workers"],
            )
            data_loaders.append(data_loader)
        
        return data_loaders 
    
    def create_dataloaders(self, ds_list):
        """Create a set of dataloaders from a list of datasets"""
        data_loaders = []
        for ds in ds_list:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["workers"],
            )
            data_loaders.append(data_loader)
        
        return data_loaders 
    
    def predict_dataloader(self, df):
        ## Validation loaders are a list https://github.com/PyTorchLightning/pytorch-lightning/issues/10809
        data_loaders = []
        ds = TreeDataset(df=df, config=self.config)        
        for x in range(len(self.models)):
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=False,
                num_workers=self.config["workers"],
            )
            data_loaders.append(data_loader)
        
        return data_loaders
        
    def configure_optimizers(self):
        """Create a optimizer for each level"""
        optimizers = []
        for x, ds in enumerate(self.train_datasets):
            optimizer = torch.optim.Adam(self.models[x].parameters(), lr=self.config["lr"])
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
        year, individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[optimizer_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[optimizer_idx])    
        self.log("train_loss_{}".format(optimizer_idx),loss, on_epoch=True, on_step=False)

        return loss        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss 
        """
        year, individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[dataloader_idx])   
        
        self.log("val_loss",loss)
        metric_dict = self.models[dataloader_idx].metrics(y_hat, y)
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        
        return {"individual":individual, "yhat":y_hat, "label":y, "year":year}  
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Calculate predictions
        """
        year, individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        y_hat = F.softmax(y_hat, dim=1)
        return year, individual, y_hat
    
    def validation_epoch_end(self, validation_step_outputs): 
        for level, results in enumerate(validation_step_outputs):
            yhat = torch.cat([x["yhat"] for x in results]).cpu().numpy()
            labels = torch.cat([x["label"] for x in results]).cpu().numpy()            
            yhat = np.argmax(yhat, 1)
            epoch_micro = torchmetrics.functional.accuracy(
                preds=torch.tensor(labels.values),
                target=torch.tensor(yhat),
                average="micro")
            
            epoch_macro = torchmetrics.functional.accuracy(
                preds=torch.tensor(labels.values),
                target=torch.tensor(yhat),
                average="macro",
                num_classes=len(self.species_label_dict)
            )
            
            self.log("Epoch Micro Accuracy", epoch_micro)
            self.log("Epoch Macro Accuracy", epoch_macro)
            
            # Log results by species
            taxon_accuracy = torchmetrics.functional.accuracy(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels.values), 
                average="none", 
                num_classes=len(self.level_label_dicts[level])
            )
            taxon_precision = torchmetrics.functional.precision(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels.values), 
                average="none", 
                num_classes=len(self.level_label_dicts[level])
            )
            species_table = pd.DataFrame(
                {"taxonID":self.level_label_dicts[level].keys(),
                 "accuracy":taxon_accuracy,
                 "precision":taxon_precision
                 })
            
            for key, value in species_table.set_index("taxonID").accuracy.to_dict().items():
                self.log("Epoch_{}_accuracy".format(key), value)

    def temporal_ensemble(self, predict_ouputs):
        individual_dict ={}
        for index, results in enumerate(predict_ouputs):
            year_yhat = torch.cat([x[1] for x in results]).cpu()
            individuals = np.concatenate([x[2] for x in results])
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
    
    def ensemble(self, results):
        """Given a multi-level model, create a final output prediction and score"""
        
        ensemble_taxonID = []
        ensemble_label = []
        ensemble_score = []
        
        for index,row in results.iterrows():
            if row["pred_taxa_top1_level_0"] == "PIPA2":
                ensemble_taxonID.append("PIPA2")
                ensemble_label.append(self.species_label_dict["PIPA2"])
                ensemble_score.append(row["top1_score_level_0"])                
            else:
                if row["pred_taxa_top1_level_1"] == "BROADLEAF":
                    if row["pred_taxa_top1_level_2"] == "OAK":
                        ensemble_taxonID.append(row["pred_taxa_top1_level_4"])
                        ensemble_label.append(self.species_label_dict[row["pred_taxa_top1_level_4"]])
                        ensemble_score.append(row["top1_score_level_4"])
                    else:
                        ensemble_taxonID.append(row["pred_taxa_top1_level_2"])
                        ensemble_label.append(self.species_label_dict[row["pred_taxa_top1_level_2"]])
                        ensemble_score.append(row["top1_score_level_2"])                     
                else:
                    ensemble_taxonID.append(row["pred_taxa_top1_level_3"])
                    ensemble_label.append(self.species_label_dict[row["pred_taxa_top1_level_3"]])
                    ensemble_score.append(row["top1_score_level_3"])
        
        results["ensembleTaxonID"] = ensemble_taxonID
        results["ens_score"] = ensemble_score
        results["ens_label"] = ensemble_label
        
        return results[["geometry","individual","ens_score","ensembleTaxonID","ens_label"]]
            
    def evaluation_scores(self, ensemble_df, experiment):   
        self.test_df["individual"] = self.test_df["individualID"]
        ensemble_df = ensemble_df.merge(self.test_df, on="individual")
        
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