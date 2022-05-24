#Multiple stage model
from functools import reduce
import geopandas as gpd
from src.models.year import learned_ensemble
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
    def __init__(self, years, classes, config):
        super().__init__()
        #Load from state dict of previous run
        self.model = learned_ensemble(classes=classes, years=years, config=config)
        
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes)
        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             })
        
    def forward(self,x):
        score = self.model(x)        
        
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
        self.level_label_dicts = []    
        self.label_to_taxonIDs = []   
        self.train_df = train_df
        self.test_df = test_df
        self.train_datasets, self.test_datasets = self.create_datasets()
        self.classes = len(self.train_df.label.unique())
        
        for index, ds in enumerate(self.train_datasets): 
            labels = torch.stack([x[2] for x in ds])
            classes = self.num_classes[index]
            base = base_model(classes=classes, years=len(self.years), config=config)
            loss_weight = []
            for x in range(classes):
                try:
                    w = 1/np.sum(labels==x)
                except:
                    w = 1 
                loss_weight.append(w)

            loss_weight = np.array(loss_weight/np.max(loss_weight))
            loss_weight[loss_weight < self.config["min_loss_weight"]] = self.config["min_loss_weight"]  
            
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
        self.num_classes = []
        self.level_id = []
        self.level_label_dicts.append({"PIPA2":0,"OTHER":1})
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[0].items()})
        
        self.level_0_train = self.train_df.copy()
        self.level_0_train.loc[~(self.level_0_train.taxonID == "PIPA2"),"taxonID"] = "OTHER"
        self.level_0_train.loc[(self.level_0_train.taxonID == "PIPA2"),"taxonID"] = "PIPA2"  
        
        #equal sample
        ids_to_keep = self.level_0_train.groupby("taxonID").apply(lambda x:x.sample(frac=1).head(self.config["PIPA2_sampling_ceiling"])).individualID
        self.level_0_train =  self.level_0_train[self.level_0_train.individualID.isin(ids_to_keep)].reset_index(drop=True)
         
        self.level_0_train["label"] = [self.level_label_dicts[0][x] for x in self.level_0_train.taxonID]
        self.level_0_train_ds = TreeDataset(df=self.level_0_train, config=self.config)
        train_datasets.append(self.level_0_train_ds)
        self.num_classes.append(len(self.level_0_train.taxonID.unique()))
        
        self.level_0_test = self.test_df.copy()
        self.level_0_test.loc[~(self.level_0_test.taxonID == "PIPA2"),"taxonID"] = "OTHER"
        self.level_0_test.loc[(self.level_0_test.taxonID == "PIPA2"),"taxonID"] = "PIPA2"                        
        self.level_0_test["label"]= [self.level_label_dicts[0][x] for x in self.level_0_test.taxonID]            
        self.level_0_test_ds = TreeDataset(df=self.level_0_test, config=self.config)
        test_datasets.append(self.level_0_test_ds)
        self.level_id.append(0)
        
        ## Level 1
        self.level_label_dicts.append({"CONIFER":0,"BROADLEAF":1})
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[1].items()})
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
        self.level_1_train_ds = TreeDataset(df=self.level_1_train, config=self.config)
        train_datasets.append(self.level_1_train_ds)
        self.num_classes.append(len(self.level_1_train.taxonID.unique()))
        
        self.level_1_test = self.test_df.copy()
        self.level_1_test = self.level_1_test[~(self.level_1_test.taxonID=="PIPA2")].reset_index(drop=True)    
        self.level_1_test.loc[~self.level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "BROADLEAF"   
        self.level_1_test.loc[self.level_1_test.taxonID.isin(["PICL","PIEL","PITA"]),"taxonID"] = "CONIFER"            
        self.level_1_test["label"] = [self.level_label_dicts[1][x] for x in self.level_1_test.taxonID]
        self.level_1_test_ds = TreeDataset(df=self.level_1_test, config=self.config)
        test_datasets.append(self.level_1_test_ds)
        self.level_id.append(1)
        
        ## Level 2
        broadleaf = [x for x in list(self.species_label_dict.keys()) if (not x in ["PICL","PIEL","PITA","PIPA2"]) & (not "QU" in x)]     
        broadleaf = {v:k for k, v in enumerate(broadleaf)}
        broadleaf["OAK"] = len(broadleaf)
        self.level_label_dicts.append(broadleaf)
        self.label_to_taxonIDs.append({v: k  for k, v in broadleaf.items()})
                    
        self.level_2_train = self.train_df.copy()
        self.level_2_train = self.level_2_train[~self.level_2_train.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])].reset_index(drop=True)
        self.level_2_train.loc[self.level_2_train.taxonID.str.contains("QU"),"taxonID"] = "OAK"
        self.level_2_train["label"] = [self.level_label_dicts[2][x] for x in self.level_2_train.taxonID]
        self.level_2_train_ds = TreeDataset(df=self.level_2_train, config=self.config)
        train_datasets.append(self.level_2_train_ds)
        self.num_classes.append(len(self.level_2_train.taxonID.unique()))
        
        self.level_2_test = self.test_df.copy()
        self.level_2_test = self.level_2_test[~self.level_2_test.taxonID.isin(["PICL","PIEL","PITA","PIPA2"])].reset_index(drop=True) 
        self.level_2_test.loc[self.level_2_test.taxonID.str.contains("QU"),"taxonID"] = "OAK"
        self.level_2_test["label"] = [self.level_label_dicts[2][x] for x in self.level_2_test.taxonID]
        self.level_2_test_ds = TreeDataset(df=self.level_2_test, config=self.config)
        test_datasets.append(self.level_2_test_ds)
        self.level_id.append(2)
        
        ## Level 3
        evergreen = [x for x in list(self.species_label_dict.keys()) if x in ["PICL","PIEL","PITA"]]         
        evergreen = {v:k for k, v in enumerate(evergreen)}
        self.level_label_dicts.append(evergreen)  
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[3].items()})
                    
        self.level_3_train = self.train_df.copy()
        self.level_3_train = self.level_3_train[self.level_3_train.taxonID.isin(["PICL","PIEL","PITA"])].reset_index(drop=True) 
        self.level_3_train["label"] = [self.level_label_dicts[3][x] for x in self.level_3_train.taxonID]
        self.level_3_train_ds = TreeDataset(df=self.level_3_train, config=self.config)
        train_datasets.append(self.level_3_train_ds)
        self.num_classes.append(len(self.level_3_train.taxonID.unique()))
        
        self.level_3_test = self.test_df.copy()
        self.level_3_test = self.level_3_test[self.level_3_test.taxonID.isin(["PICL","PIEL","PITA"])].reset_index(drop=True) 
        self.level_3_test["label"] = [self.level_label_dicts[3][x] for x in self.level_3_test.taxonID]
        self.level_3_test_ds = TreeDataset(df=self.level_3_test, config=self.config)
        test_datasets.append(self.level_3_test_ds)
        self.level_id.append(3)
        
        ## Level 4
        oak = [x for x in list(self.species_label_dict.keys()) if "QU" in x]
        self.level_label_dicts.append({v:k for k, v in enumerate(oak)})
        self.label_to_taxonIDs.append({v: k  for k, v in self.level_label_dicts[4].items()})
        
        #Balance the train in OAKs
        self.level_4_train = self.train_df.copy()
        self.level_4_train = self.level_4_train[self.level_4_train.taxonID.str.contains("QU")].reset_index(drop=True)
        self.level_4_train["label"] = [self.level_label_dicts[4][x] for x in self.level_4_train.taxonID]
        
        ids_to_keep = self.level_4_train.groupby("taxonID").apply(lambda x:x.sample(frac=1).head(self.config["oaks_sampling_ceiling"])).individualID
        self.level_4_train = self.level_4_train[self.level_4_train.individualID.isin(ids_to_keep)].reset_index(drop=True)
        
        self.level_4_train_ds = TreeDataset(df=self.level_4_train, config=self.config)
        train_datasets.append(self.level_4_train_ds)
        self.num_classes.append(len(self.level_4_train.taxonID.unique()))

        self.level_4_test = self.test_df.copy()
        self.level_4_test = self.level_4_test[self.level_4_test.taxonID.str.contains("QU")].reset_index(drop=True)
        self.level_4_test["label"] = [self.level_label_dicts[4][x] for x in self.level_4_test.taxonID]
        self.level_4_test_ds = TreeDataset(df=self.level_4_test, config=self.config)
        test_datasets.append(self.level_4_test_ds)
        self.level_id.append(4)

        return train_datasets, test_datasets
    
    def train_dataloader(self):
        data_loaders = []
        for ds in self.train_datasets:
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                shuffle=True,
                num_workers=self.config["workers"],
            )
            data_loaders.append(data_loader)
        
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
    
    def predict_dataloader(self, ds_list):
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
        individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[optimizer_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[optimizer_idx])    
        self.log("train_loss_{}".format(optimizer_idx),loss, on_epoch=True, on_step=False)

        return loss        
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss 
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=self.loss_weights[dataloader_idx])   
        
        self.log("val_loss",loss)
        metric_dict = self.models[dataloader_idx].metrics(y_hat, y)
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        
        return {"individual":individual, "yhat":y_hat, "label":y}  
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        """Calculate predictions
        """
        individual, inputs = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        y_hat = F.softmax(y_hat, dim=1)
        
        return individual, y_hat
    
    def validation_epoch_end(self, validation_step_outputs): 
        for level, results in enumerate(validation_step_outputs):
            yhat = torch.cat([x["yhat"] for x in results]).cpu().numpy()
            labels = torch.cat([x["label"] for x in results]).cpu().numpy()            
            yhat = np.argmax(yhat, 1)
            epoch_micro = torchmetrics.functional.accuracy(
                preds=torch.tensor(labels),
                target=torch.tensor(yhat),
                average="micro")
            
            epoch_macro = torchmetrics.functional.accuracy(
                preds=torch.tensor(labels),
                target=torch.tensor(yhat),
                average="macro",
                num_classes=len(self.species_label_dict)
            )
            
            self.log("Epoch Micro Accuracy", epoch_micro)
            self.log("Epoch Macro Accuracy", epoch_macro)
            
            # Log results by species
            taxon_accuracy = torchmetrics.functional.accuracy(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels), 
                average="none", 
                num_classes=len(self.level_label_dicts[level])
            )
            taxon_precision = torchmetrics.functional.precision(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels), 
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
    
    def gather_predictions(self, predict_df, return_features=False):
        """Post-process the predict method to create metrics"""
        if return_features: 
            features = []
            for level in predict_df:
                features.append(np.vstack(level[1]))             
            return features
        
        level_results = []
        for index, level in enumerate(predict_df):
            # Concat batches            
            individuals = np.concatenate([x[0] for x in level])
            predictions = np.concatenate([x[1] for x in level])
            
            #Create dataframe
            predictions_top1 = np.argmax(predictions, 1)    
            #predictions_top2 = pd.DataFrame(predictions).apply(lambda x: np.argsort(x.values)[-2], axis=1)
            top1_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[0], axis=1)
            #top2_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[1], axis=1)
            
            df = pd.DataFrame({
                "pred_label_top1_level_{}".format(index):predictions_top1,
                #"pred_label_top2_level_{}".format(index):predictions_top2,
                "top1_score_level_{}".format(index):top1_score,
                #"top2_score_level_{}".format(index):top2_score,
                "individual":individuals
            })
            df["pred_taxa_top1_level_{}".format(index)] = df["pred_label_top1_level_{}".format(index)].apply(lambda x: self.label_to_taxonIDs[index][x]) 
            #df["pred_taxa_top2_level_{}".format(index)] = df["pred_label_top2_level_{}".format(index)].apply(lambda x: self.label_to_taxonIDs[index][x])        
            level_results.append(df)
        
        results = reduce(lambda  left,right: pd.merge(left,right,on=['individual'],
                                                        how='outer'), level_results)                    
        return results
    
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
        
        return results[["individualID","ens_score","ensembleTaxonID","ens_label"]]
            
    def evaluation_scores(self, ensemble_df, experiment):   
        ensemble_df = ensemble_df.merge(self.test_df, on="individualID")
        
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