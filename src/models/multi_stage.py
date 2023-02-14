#Multiple stage model
from src.models.year import learned_ensemble
from src.data import TreeDataset
from src import sampler

from pytorch_lightning import LightningModule
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torchmetrics
from torchmetrics import Accuracy, ClasswiseWrapper, Precision, MetricCollection
import torch
import traceback
import warnings

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
    def __init__(self, train_df, test_df, config, train_mode=True, debug=False):
        super().__init__()
        # Generate each model
        self.years = train_df.tile_year.unique()
        self.config = config
        self.models = nn.ModuleDict()
        self.species_label_dict = train_df[["taxonID","label"]].drop_duplicates().set_index("taxonID").to_dict()["label"]
        self.index_to_label = {v:k for k,v in self.species_label_dict.items()}
        self.level_label_dicts = []    
        self.label_to_taxonIDs = []   
        self.train_df = train_df
        self.test_df = test_df
        
        #Lookup taxonomic names
        self.taxonomy = pd.read_csv(config["taxonomic_csv"])
        
        #remove anything not current in taxonomy and warn
        missing_ids = self.train_df.loc[~self.train_df.taxonID.isin(self.taxonomy.taxonID)].taxonID.unique()
        warnings.warn("The following ids are not in the taxonomy: {}!".format(missing_ids))
        self.train_df = self.train_df[~self.train_df.taxonID.isin(missing_ids)]
        self.test_df= self.test_df[~self.test_df.taxonID.isin(missing_ids)]
        
        #hotfix for old naming schema
        try:
            self.test_df["individual"] = self.test_df["individualID"]
            self.train_df["individual"] = self.train_df["individualID"]
        except:
            pass
        
        if train_mode:
            # Create the hierarchical structure
            self.train_datasets, self.train_dataframes, self.level_label_dicts = self.create_datasets(self.train_df)
            self.test_datasets, self.test_dataframes, _ = self.create_datasets(self.test_df)
            
            #Create label dicts
            self.label_to_taxonIDs = {}
            for x in self.level_label_dicts:
                self.label_to_taxonIDs[x] = {v: k  for k, v in self.level_label_dicts[x].items()}
            
            self.levels = len(self.train_datasets)       
            self.level_names = list(self.train_datasets.keys())
            
            #Generate metrics for each class level
            self.level_metrics = nn.ModuleDict()
            for key,value in self.test_datasets.items():
                taxon_level_labels = list(self.level_label_dicts[key].keys())
                num_classes = len(self.level_label_dicts[key])
                level_metric = MetricCollection({       
                "Species accuracy":ClasswiseWrapper(Accuracy(average="none", num_classes=num_classes), labels=taxon_level_labels),
                "Species precision":ClasswiseWrapper(Precision(average="none", num_classes=num_classes),labels=taxon_level_labels),
                })
                self.level_metrics[key] = level_metric

            self.classes = len(self.train_df.label.unique())
            for key, value in self.train_dataframes.items(): 
                classes = len(value.label.unique())
                base = base_model(classes=classes, years=len(self.years), config=self.config)
                self.models[key] = base           
            if not debug:
                self.save_hyperparameters()        
    
    def dominant_class_model(self, df, level_label_dict=None):
        """A level 0 model splits out the dominant class and compares to all other samples"""
        
        if level_label_dict is None:
            common_species = df.taxonID.value_counts().reset_index()
            common_species = common_species[common_species.taxonID > self.config["head_class_minimum_samples"]]["index"]

            if common_species.empty:
                raise ValueError("No data with samples more than {} samples".format(self.config["head_class_minimum_samples"]))
                    
            level_label_dict = {value:key for key, value in enumerate(common_species)}
            level_label_dict["CONIFER"] = len(level_label_dict)
            level_label_dict["BROADLEAF"] = len(level_label_dict)
        else:
            common_species = list(level_label_dict.keys())
        
        # Select head and tail classes
        head_classes = df[df.taxonID.isin(common_species)]
        
        #Split tail classes into conifer and broadleaf
        tail_classes = df[~df.taxonID.isin(common_species)]
        needleleaf = self.taxonomy[self.taxonomy.families=="Pinidae"].taxonID
        needleleaf = needleleaf[~needleleaf.isin(common_species)]
        broadleaf = self.taxonomy[~(self.taxonomy.families=="Pinidae")].taxonID
        broadleaf = broadleaf[~broadleaf.isin(common_species)]
        tail_classes.loc[tail_classes.taxonID.isin(needleleaf),"taxonID"] = "CONIFER"
        tail_classes.loc[tail_classes.taxonID.isin(broadleaf),"taxonID"] = "BROADLEAF"
        
        # Create labels
        level_0 = pd.concat([head_classes, tail_classes])                
        level_0["label"] = [level_label_dict[x] for x in level_0.taxonID]
        level_0_ds = TreeDataset(df=level_0, config=self.config)
        
        return level_0_ds, level_0, level_label_dict
    
    def conifer_model(self, df, level_label_dict=None):
        needleleaf = self.taxonomy[self.taxonomy.families=="Pinidae"].taxonID        
        needleleaf = [x for x in df.taxonID.unique() if x in needleleaf.values]
        common_species = df.taxonID.value_counts().reset_index()
        common_species = common_species[common_species.taxonID > self.config["head_class_minimum_samples"]]["index"]
        needleleaf = [x for x in needleleaf if not x in common_species]
        
        if len(needleleaf) < 2:
            raise ValueError("Not enough conifer species")
        
        level_label_dict = {value:key for key, value in enumerate(needleleaf)}
                    
        # Select head and tail classes
        level_1 = df[df.taxonID.isin(needleleaf)]
        
        # Create labels
        level_1["label"] = [level_label_dict[x] for x in level_1.taxonID]
        level_1_ds = TreeDataset(df=level_1, config=self.config)
        
        return level_1_ds, level_1, level_label_dict
    
    def broadleaf_model(self, df, level_label_dict=None):
        """Model for the broadleaf species"""
        broadleaf = self.taxonomy[~(self.taxonomy.families=="Pinidae")].taxonID
        common_species = df.taxonID.value_counts().reset_index()
        common_species = common_species[common_species.taxonID > self.config["head_class_minimum_samples"]]["index"]        
        broadleaf = broadleaf[~broadleaf.isin(common_species)]
        broadleaf_species = [x for x in df.taxonID.unique() if x in broadleaf.values]        
        level_2 = df[df.taxonID.isin(broadleaf_species)]
        
        if len(broadleaf_species) < 2:
            raise ValueError("Not enough broadleaf species")
        
        non_oak = [x for x in broadleaf_species if not x[0:2] == "QU"]
        level_label_dict = {value:key for key, value in enumerate(non_oak)}
        
        if len(non_oak) > 2:
            oak = [x for x in broadleaf_species if x[0:2] == "QU"]
            level_label_dict["OAK"] = len(level_label_dict) 
            level_2.loc[level_2.taxonID.isin(oak),"taxonID"] = "OAK"
            
        # Select head and tail classes
        level_2["label"] = [level_label_dict[x] for x in level_2.taxonID]
        level_2_ds = TreeDataset(df=level_2, config=self.config)
        
        return level_2_ds, level_2, level_label_dict
        
    def oak_model(self, df, level_label_dict=None):
        broadleaf = self.taxonomy[~(self.taxonomy.families=="Pinidae")].taxonID
        common_species = df.taxonID.value_counts().reset_index()
        common_species = common_species[common_species.taxonID > self.config["head_class_minimum_samples"]]["index"]        
        broadleaf = broadleaf[~broadleaf.isin(common_species)]
        broadleaf_species = [x for x in df.taxonID.unique() if x in broadleaf.values]        
        oak = [x for x in broadleaf_species if x[0:2] == "QU"]
        level_3 = df[df.taxonID.isin(broadleaf_species)]
        
        if not len(oak) > 2:
            raise ValueError("Not enough Oak Species")
        
        level_label_dict = {value:key for key, value in enumerate(oak)}
        level_label_dict["OAK"] = len(level_label_dict) 
            
        # Select head and tail classes
        level_3 = level_3[level_3.taxonID.isin(oak)]
        level_3["label"] = [level_label_dict[x] for x in level_3.taxonID]
        level_3_ds = TreeDataset(df=level_3, config=self.config)
        
        return level_3_ds, level_3, level_label_dict
                
    def create_datasets(self, df, level_label_dicts=None):
        """Create a hierarchical set of dataloaders by splitting into taxonID groups
        train: whether to sample the training labels
        """
        datasets = {}
        dataframes = {}
        
        if level_label_dicts is None:
            level_label_dicts = {}
            level_label_dicts["dominant_class"] = None
            level_label_dicts["conifer"] = None
            level_label_dicts["broadleaf"] = None
            level_label_dicts["oak"] = None
            
        try:
            level_ds, level_df, level_label_dict = self.dominant_class_model(df=df, level_label_dict=level_label_dicts["dominant_class"])
            level_label_dicts["dominant_class"] = level_label_dict
            dataframes["dominant_class"] = level_df
            datasets["dominant_class"] = level_ds            
        except ValueError:
            print("No data with samples more than {} samples".format(self.config["head_class_minimum_samples"]))
            traceback.print_exc()            
        try:
            level_ds, level_df, level_label_dict = self.conifer_model(df, level_label_dict=level_label_dicts["conifer"])
            level_label_dicts["conifer"] = level_label_dict
            dataframes["conifer"] = level_df
            datasets["conifer"] = level_ds           
        except ValueError:
            print("Conifer model failed")
            traceback.print_exc()
        
        try:
            level_ds, level_df, level_label_dict = self.broadleaf_model(df, level_label_dict=level_label_dicts["broadleaf"])
            level_label_dicts["broadleaf"] = level_label_dict
            dataframes["broadleaf"] = level_df
            datasets["broadleaf"] = level_ds           
        except ValueError:
            print("broadleaf model failed")
            traceback.print_exc()                
        try:
            level_ds, level_df, level_label_dict = self.oak_model(df, level_label_dict=level_label_dicts["oak"])
            level_label_dicts["oak"] = level_label_dict
            dataframes["oak"] = level_df
            datasets["oak"] = level_ds           
        except ValueError:
            print("Oak model failed")
            traceback.print_exc()
        
        #Delete any empty keys
        level_label_dicts = {k: v for k, v in self.level_label_dicts.items() if v}
        
        return datasets, dataframes, level_label_dicts
    
    def train_dataloader(self):
        data_loaders = []
                
        for index, ds in self.train_datasets.items():
            #Multi-balance
            imbalance = self.train_dataframes[index]
            one_hot = torch.nn.functional.one_hot(torch.tensor(imbalance.groupby("individual", sort=False).apply(lambda x: x.head(1)).label.values))
            train_sampler = sampler.MultilabelBalancedRandomSampler(
                labels=one_hot, indices=range(len(imbalance.individual.unique())), class_choice="cycle")
            
            data_loader = torch.utils.data.DataLoader(
                ds,
                batch_size=self.config["batch_size"],
                sampler=train_sampler,
                num_workers=self.config["workers"]
            )
            data_loaders.append(data_loader)
        
        return data_loaders        

    def val_dataloader(self):
        data_loaders = []
        for ds in self.test_datasets:
            data_loader = torch.utils.data.DataLoader(
                self.test_datasets[ds],
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
        for x, ds in self.train_datasets.items():
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
            
            optimizers.append({'optimizer':optimizer, 'lr_scheduler': {"scheduler":scheduler, "monitor":'val_loss_{}'.format(x), "frequency":self.config["validation_interval"], "interval":"epoch"}})

        return optimizers     
        
    def training_step(self, batch, batch_idx, optimizer_idx):
        """Calculate train_df loss
        """
        level_name = self.level_names[optimizer_idx]
        individual, inputs, y = batch[optimizer_idx]
        images = inputs["HSI"]  
        y_hat = self.models[level_name].forward(images)
        loss = F.cross_entropy(y_hat, y)    
        self.log("train_loss_{}".format(level_name),loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Calculate val loss and on_epoch metrics
        """
        level_name = self.level_names[dataloader_idx]        
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[level_name].forward(images)
        loss = F.cross_entropy(y_hat, y)   
        
        self.log("val_loss_{}".format(level_name),loss,add_dataloader_idx=False)
        try:
            self.models[level_name].metrics(y_hat, y)
        except:
            print("Validation failed with targets {}".format(y))
            
        self.log_dict(self.models[level_name].metrics, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        
        self.level_metrics[level_name].update(y_hat, y)
 
        return {"individual":individual, "yhat":y_hat, "label":y}  
    
    def predict_step(self, batch, batch_idx):
        """Calculate predictions
        """
        individual, inputs = batch
        images = inputs["HSI"]  
        
        y_hats = []
        for level in self.models:   
            y_hat = self.models[level].forward(images)
            y_hat = F.softmax(y_hat, dim=1)
            y_hats.append(y_hat)
        
        return individual, y_hats
    
    def on_predict_epoch_end(self, outputs):
        outputs = self.all_gather(outputs)
    
    def on_validation_epoch_end(self):
        for level, ds in self.test_datasets.items():
            class_metrics = self.level_metrics[level].compute()
            self.log_dict(class_metrics, on_epoch=True, on_step=False)
            self.level_metrics[level].reset()
        
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
                    levels.append(self.level_names[index])
                
        temporal_average = pd.DataFrame({"individual":individuals,"level":levels,"yhat":yhats})
                
        #Argmax and score for each level
        predicted_label = temporal_average.groupby(["individual","level"]).yhat.apply(
            lambda x: np.argmax(np.vstack(x))).reset_index().pivot(
                index=["individual"],columns="level",values="yhat").reset_index()
        
        predicted_score = temporal_average.groupby(["individual","level"]).yhat.apply(
            lambda x: np.vstack(x).max()).reset_index().pivot(
                index=["individual"],columns="level",values="yhat").reset_index()
        
        results = pd.merge(predicted_label,predicted_score, on="individual")
        
        #clean up column names from merge
        results.columns = results.columns.str.replace("x","label")
        results.columns = results.columns.str.replace("y","score")
        
        #Label taxa
        for level, label_dict in self.label_to_taxonIDs.items():
            results["{}_taxa".format(level)] = results["{}_label".format(level)].apply(lambda x: label_dict[x])
        
        return results
    
    def ensemble(self, results):
        """Given a multi-level model, create a final output prediction and score"""
        ensemble_taxonID = []
        ensemble_label = []
        ensemble_score = []
        
        #For each level, select the predicted taxonID and retrieve the original label order
        for index,row in results.iterrows():
            if not row["dominant_class_taxa"] in ["CONIFER","BROADLEAF"]:
                ensemble_taxonID.append(row["dominant_class_taxa"])
                ensemble_label.append(self.species_label_dict[row["dominant_class_taxa"]])
                ensemble_score.append(row["dominant_class_score"])                
            elif row["dominant_class_taxa"] =="CONIFER":
                ensemble_taxonID.append(row["conifer_taxa"])
                ensemble_label.append(self.species_label_dict[row["conifer_taxa"]])
                ensemble_score.append(row["conifer_score"])                   
            elif not row["broadleaf_taxa"] =="OAK":
                ensemble_taxonID.append(row["broadleaf_taxa"])
                ensemble_label.append(self.species_label_dict[row["broadleaf_taxa"]])
                ensemble_score.append(row["broadleaf_score"])
            else:
                ensemble_taxonID.append(row["oak_taxa"])
                ensemble_label.append(self.species_label_dict[row["oak_taxa"]])
                ensemble_score.append(row["oak_taxa"])                
                
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
