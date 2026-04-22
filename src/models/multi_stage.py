#Multiple stage model
from functools import reduce
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
import torch

class base_model(Module):
    def __init__(self, years, classes, config):
        super().__init__()
        #Load from state dict of previous run
        self.model = learned_ensemble(classes=classes, years=years, config=config)
        
        micro_recall = torchmetrics.Accuracy(task="multiclass", num_classes=classes, average="micro")
        macro_recall = torchmetrics.Accuracy(task="multiclass", num_classes=classes, average="macro")
        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             })
        
    def forward(self,x):
        score = self.model(x)        
        
        return score 
    
class MultiStage(LightningModule):
    """Hierarchical ensemble; uses manual optimization (PyTorch Lightning 2+)."""

    automatic_optimization = False

    def __init__(self, train_df, test_df, crowns, config, train_mode=True):
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
            self._val_epoch_outputs = []
        
            self.classes = len(self.train_df.label.unique())
            for index, ds in enumerate([self.level_0_train, self.level_1_train, self.level_2_train, self.level_3_train, self.level_4_train]): 
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
            self.save_hyperparameters()        
            
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
        PIPA2 = self.level_0_train[self.level_0_train.taxonID=="PIPA2"]
        nonPIPA2 = self.level_0_train[~(self.level_0_train.taxonID=="PIPA2")]
        nonPIPA2ids = (
            nonPIPA2.groupby("individual", as_index=False)
            .head(1)
            .groupby("taxonID", as_index=False)
            .head(self.config["other_sampling_ceiling"])
        )["individual"]
        nonPIPA2 = nonPIPA2[nonPIPA2.individual.isin(nonPIPA2ids)]
        self.level_0_train = pd.concat([PIPA2, nonPIPA2])
        self.level_0_train.loc[~(self.level_0_train.taxonID == "PIPA2"),"taxonID"] = "OTHER"
                
        self.level_0_train["label"] = [self.level_label_dicts[0][x] for x in self.level_0_train.taxonID]
        self.level_0_train_ds = TreeDataset(df=self.level_0_train, config=self.config)
        train_datasets.append(self.level_0_train_ds)
        self.num_classes.append(len(self.level_0_train.taxonID.unique()))
        
        self.level_0_test = self.test_df.copy()
        self.level_0_test.loc[~(self.level_0_test.taxonID == "PIPA2"),"taxonID"] = "OTHER"
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
        conifer_ids = self.level_1_train[self.level_1_train.taxonID == "CONIFER"]["individual"].to_numpy()
        broadleaf_parts = []
        cap = math.ceil(len(conifer_ids) / 11)
        for _, g in self.level_1_train[self.level_1_train.taxonID == "BROADLEAF"].groupby("label"):
            g = g.sample(frac=1)
            g = g.groupby("individual", as_index=False).head(1)
            broadleaf_parts.append(g.head(cap))
        broadleaf_ids = pd.concat(broadleaf_parts, ignore_index=True)["individual"].to_numpy()
        ids_to_keep = np.concatenate([broadleaf_ids, conifer_ids])
        self.level_1_train = self.level_1_train[self.level_1_train.individual.isin(ids_to_keep)].reset_index(drop=True)
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
        
        non_oakid = self.level_2_train[~(self.level_2_train.taxonID == "OAK")]["individual"].to_numpy()
        oak_parts = []
        oak_cap = int(len(non_oakid) / 5)
        for _, g in self.level_2_train[self.level_2_train.taxonID == "OAK"].groupby("label"):
            oak_parts.append(g.sample(frac=1).head(oak_cap))
        oak_ids = pd.concat(oak_parts, ignore_index=True)["individual"].to_numpy()
        ids_to_keep = np.concatenate([oak_ids, non_oakid])
        self.level_2_train = self.level_2_train[self.level_2_train.individual.isin(ids_to_keep)].reset_index(drop=True)
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
        self.level_3_train = self.level_3_train.groupby(
            "taxonID", group_keys=False
        ).head(self.config["evergreen_ceiling"])
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
        oak_parts = []
        for _, g in self.level_4_train.groupby("taxonID"):
            g = g.sample(frac=1)
            g = g.groupby("individual", as_index=False).head(1)
            oak_parts.append(g.head(self.config["oaks_sampling_ceiling"]))
        ids_to_keep = pd.concat(oak_parts, ignore_index=True)["individual"].to_numpy()
        self.level_4_train = self.level_4_train[self.level_4_train.individual.isin(ids_to_keep)].reset_index(drop=True)
        
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
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.75,
                patience=8,
                threshold=0.0001,
                threshold_mode="rel",
                cooldown=0,
                eps=1e-08,
            )
            
            optimizers.append({'optimizer':optimizer, 'lr_scheduler': {"scheduler":scheduler, "monitor":'val_loss/dataloader_idx_{}'.format(x)}})

        return optimizers

    def on_validation_epoch_start(self):
        self._val_epoch_outputs = []

    def training_step(self, batch, batch_idx):
        """One combined batch (tuple/list) per step; optimize each level separately."""
        optimizers = self.optimizers()
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        total_loss = None
        for optimizer_idx, opt in enumerate(optimizers):
            opt.zero_grad(set_to_none=True)
            loss_weights = self.__getattr__("loss_weight_" + str(optimizer_idx))
            individual, inputs, y = batch[optimizer_idx]
            images = inputs["HSI"]
            y_hat = self.models[optimizer_idx].forward(images)
            loss = F.cross_entropy(y_hat, y, weight=loss_weights)
            self.manual_backward(loss)
            opt.step()
            self.log("train_loss_{}".format(optimizer_idx), loss, on_epoch=True, on_step=False)
            total_loss = loss if total_loss is None else total_loss + loss
        return total_loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss 
        """
        loss_weight = self.__getattr__('loss_weight_'+str(dataloader_idx))        
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[dataloader_idx].forward(images)
        loss = F.cross_entropy(y_hat, y, weight=loss_weight)   
        
        self.log("val_loss",loss)
        metric_dict = self.models[dataloader_idx].metrics(y_hat, y)
        self.log_dict(metric_dict, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        self._val_epoch_outputs.append((dataloader_idx, {"yhat": y_hat, "label": y}))

        return loss

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
    
    def on_predict_epoch_end(self, *args, **kwargs):
        return
        
    def on_validation_epoch_end(self):
        if not self._val_epoch_outputs:
            return
        by_loader = {}
        for idx, o in self._val_epoch_outputs:
            by_loader.setdefault(idx, []).append(o)
        for level, results in by_loader.items():
            yhat = torch.cat([x["yhat"] for x in results]).cpu().numpy()
            labels = torch.cat([x["label"] for x in results]).cpu().numpy()            
            yhat = np.argmax(yhat, 1)
            n_cls = len(self.species_label_dict)
            epoch_micro = torchmetrics.functional.accuracy(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels),
                task="multiclass",
                num_classes=n_cls,
                average="micro",
            )

            epoch_macro = torchmetrics.functional.accuracy(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels),
                task="multiclass",
                num_classes=n_cls,
                average="macro",
            )
            
            self.log("Epoch Micro Accuracy level {}".format(level), epoch_micro)
            self.log("Epoch Macro Accuracy level {}".format(level), epoch_macro)
            
            # Log results by species
            n_level = len(self.level_label_dicts[level])
            taxon_accuracy = torchmetrics.functional.accuracy(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels),
                task="multiclass",
                num_classes=n_level,
                average="none",
            )
            taxon_precision = torchmetrics.functional.precision(
                preds=torch.tensor(yhat),
                target=torch.tensor(labels),
                task="multiclass",
                num_classes=n_level,
                average="none",
            )
            species_table = pd.DataFrame(
                {"taxonID":self.level_label_dicts[level].keys(),
                 "accuracy":taxon_accuracy,
                 "precision":taxon_precision
                 })
            
            for key, value in species_table.set_index("taxonID").accuracy.to_dict().items():
                self.log("Epoch_{}_accuracy".format(key), value)
    
            for key, value in species_table.set_index("taxonID").precision.to_dict().items():
                self.log("Epoch_{}_precision".format(key), value)

        self._val_epoch_outputs = []

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # PyTorch 2.6+ defaults to weights-only unpickling; Lightning checkpoints need full load.
        kwargs.setdefault("weights_only", False)
        return super(MultiStage, cls).load_from_checkpoint(checkpoint_path, **kwargs)

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
        predicted_label.columns = ["individual","pred_label_top1_level_0","pred_label_top1_level_1",
                                   "pred_label_top1_level_2","pred_label_top1_level_3","pred_label_top1_level_4"]
        
        predicted_score = temporal_average.groupby(["individual","level"]).yhat.apply(
            lambda x: np.vstack(x).max()).reset_index().pivot(
                index=["individual"],columns="level",values="yhat").reset_index()
        predicted_score.columns = ["individual","top1_score_level_0","top1_score_level_1",
                                   "top1_score_level_2","top1_score_level_3","top1_score_level_4"]
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
        
        return results
            
    def evaluation_scores(self, ensemble_df, experiment):   
        ensemble_df = ensemble_df.drop_duplicates(subset=["individual"], keep="first")
        n_cls = len(self.species_label_dict)
        ed = ensemble_df.dropna(subset=["ens_label", "label"])
        ed = ed[
            (ed["ens_label"] >= 0)
            & (ed["ens_label"] < n_cls)
            & (ed["label"] >= 0)
            & (ed["label"] < n_cls)
        ]
        if ed.empty:
            return ensemble_df

        preds = torch.tensor(ed["ens_label"].values, dtype=torch.long)
        target = torch.tensor(ed["label"].values, dtype=torch.long)
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=preds,
            target=target,
            task="multiclass",
            num_classes=n_cls,
            average="none",
        )

        taxon_precision = torchmetrics.functional.precision(
            preds=preds,
            target=target,
            task="multiclass",
            num_classes=n_cls,
            average="none",
        )

        taxon_labels = sorted(self.species_label_dict.keys(), key=lambda t: self.species_label_dict[t])
        species_table = pd.DataFrame(
            {"taxonID": taxon_labels,
             "accuracy": taxon_accuracy,
             "precision": taxon_precision,
             })
        
        if experiment:
            experiment.log_metrics(species_table.set_index("taxonID").accuracy.to_dict(),prefix="accuracy")
            experiment.log_metrics(species_table.set_index("taxonID").precision.to_dict(),prefix="precision")
                
        # Log result by site
        if experiment:
            site_data_frame =[]
            for name, group in ed.groupby("siteID"):
                g = group[
                    (group["ens_label"] >= 0)
                    & (group["ens_label"] < n_cls)
                    & (group["label"] >= 0)
                    & (group["label"] < n_cls)
                ]
                if g.empty:
                    continue
                site_micro = np.sum(g.ens_label.values == g.label.values) / len(g.ens_label.values)
                
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(g["ens_label"].values, dtype=torch.long),
                    target=torch.tensor(g["label"].values, dtype=torch.long),
                    task="multiclass",
                    num_classes=n_cls,
                    average="macro",
                )
                                
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro) 
                
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro], "Macro Recall": [site_macro]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)        
        
        return ensemble_df