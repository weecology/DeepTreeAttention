#Multiple stage model
import os
import warnings
import traceback

from src.models.year import learned_ensemble
from src import sampler, utils, augmentation

from pytorch_lightning import LightningModule
import pandas as pd
import numpy as np
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torchmetrics
from torchmetrics import Accuracy, ClasswiseWrapper, Precision, MetricCollection
import torch
from torch.utils.data import Dataset


# Dataset class
class TreeDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
    """
    def __init__(self, df=None, csv_file=None, config=None, train=True, image_dict=None):
        if csv_file:
            self.annotations = pd.read_csv(csv_file)
        else:
            self.annotations = df
        
        self.train = train
        self.config = config         
        self.image_size = config["image_size"]
        self.years = self.annotations.tile_year.unique()
        self.individuals = self.annotations.individual.unique()
        self.image_paths = self.annotations.groupby("individual").apply(lambda x: x.set_index('tile_year').image_path.to_dict())
        self.image_dict = image_dict
        if train:
            self.labels = self.annotations.set_index("individual").label.to_dict()
        
        # Create augmentor
        self.transformer = augmentation.augment(image_size=self.image_size, train=train, pad_or_resize=config["pad_or_resize"])
                     
        # Pin data to memory if desired 
        if self.config["preload_images"]:
            if self.image_dict is None:
                self.image_dict = { }
                for individual in self.individuals:
                    images = { }
                    ind_annotations = self.image_paths[individual]
                    for year in self.years:
                        try:
                            year_annotations = ind_annotations[year]
                        except KeyError:
                            images[str(year)] = image = torch.zeros(self.config["bands"], self.image_size, self.image_size)  
                            continue
                        image_path = os.path.join(self.config["crop_dir"], year_annotations)
                        image = utils.load_image(image_path)                        
                        images[str(year)] = image
                    self.image_dict[individual] = images
            
    def __len__(self):
        # 0th based index
        return len(self.individuals)

    def __getitem__(self, index):
        inputs = {}
        individual = self.individuals[index]      
        if self.config["preload_images"]:
            images = self.image_dict[individual]
        else:
            images = { }
            ind_annotations = self.image_paths[individual]
            for year in self.years:
                try:
                    year_annotations = ind_annotations[year]
                except KeyError:
                    images[str(year)] = image = torch.zeros(self.config["bands"], self.image_size, self.image_size)  
                    continue
                image_path = os.path.join(self.config["crop_dir"], year_annotations)
                try:
                    image = utils.load_image(image_path)
                except ValueError:
                    return None

                images[str(year)] = image
                
        images = {key: self.transformer(value) for key, value in images.items()}
        inputs["HSI"] = images

        if self.train:
            label = self.labels[individual]
            label = torch.tensor(label, dtype=torch.long)

            return individual, inputs, label
        else:
            return individual, inputs
        
class base_model(Module):
    def __init__(self, years, classes, config, name):
        super().__init__()
        #Load from state dict of previous run
        self.model = learned_ensemble(classes=classes, years=years, config=config)
        micro_recall = Accuracy(average="micro", task="multiclass", num_classes=classes)
        macro_recall = Accuracy(average="macro", num_classes=classes,task="multiclass")
        self.metrics = MetricCollection(
            {"Micro Accuracy_{}".format(name):micro_recall,
             "Macro Accuracy_{}".format(name):macro_recall,
             })
        
    def forward(self,x):
        score = self.model(x)        
        
        return score 
    
class MultiStage(LightningModule):
    def __init__(self, config, train_df=None, test_df=None, debug=False):
        super().__init__()
        # Generate each model and the containers for metadata
        self.config = config
        self.level_label_dicts = []    
        self.label_to_taxonIDs = []   
        self.train_df = train_df
        self.test_df = test_df
        self.current_level = None
        self.models = nn.ModuleDict()        
            
    def on_save_checkpoint(self, checkpoint):
        checkpoint["train_df"] = self.train_df
        checkpoint["test_df"] = self.test_df
    
    def on_load_checkpoint(self, checkpoint):
        self.train_df = checkpoint["train_df"] 
        self.test_df = checkpoint["test_df"] 
        
        #Don't preload images, since the dataset might not be around anymore.
        self.config["preload_image_dict"] = False
        self.config["preload_images"] = False
        self.config["head_class_minimum_ratio"] = self.config["head_class_minimum_ratio"]
        
        # Recreate models
        self.setup("fit")
        
    def setup(self, stage):
        """Setup a nested set of modules if needed"""
        if len(self.models.keys()) == 0:
            
            if self.train_df is None:
                raise ValueError("Fitting, but no train_df specificied to MultiStage.init()")
            
            self.species_label_dict = self.test_df[["taxonID","label"]].drop_duplicates().set_index("taxonID").to_dict()["label"]
            self.index_to_label = {v:k for k,v in self.species_label_dict.items()}
            self.years = self.train_df.tile_year.unique()
            
            # Lookup taxonomic names
            self.taxonomy = pd.read_csv(self.config["taxonomic_csv"])
            common_species = self.train_df.taxonID.value_counts()/self.train_df.shape[0]
            self.common_species = common_species[common_species > self.config["head_class_minimum_ratio"]].index.values

            conifer = self.taxonomy[(self.taxonomy.families=="Pinidae")].taxonID        
            conifer_species = [x for x in self.train_df.taxonID.unique() if x in conifer.values] 
            self.conifer_species = [x for x in conifer_species if not x in self.common_species]
            broadleaf = self.taxonomy[~(self.taxonomy.families=="Pinidae")].taxonID        
            broadleaf_species = [x for x in self.train_df.taxonID.unique() if x in broadleaf.values] 
            self.broadleaf_species = [x for x in broadleaf_species if not x in self.common_species]
            
            #remove anything not current in taxonomy and warn
            # missing_ids = self.train_df.loc[~self.train_df.taxonID.isin(self.taxonomy.taxonID)].taxonID.unique()
            # warnings.warn("The following ids are not in the taxonomy: {}!".format(missing_ids))
            # self.train_df = self.train_df[~self.train_df.taxonID.isin(missing_ids)]
            # self.test_df= self.test_df[~self.test_df.taxonID.isin(missing_ids)]
                    
            if self.config["preload_image_dict"]:
                self.train_image_dict = utils.preload_image_dict(self.train_df, self.config)
                self.test_image_dict = utils.preload_image_dict(self.test_df, self.config)
            else:
                self.train_image_dict = None
                self.test_image_dict = None
                
            # Create the hierarchical structure
            self.train_datasets, self.train_dataframes, self.level_label_dicts = self.create_datasets(self.train_df, image_dict=self.train_image_dict, max_samples_per_class=self.config["max_samples_per_class"])
            self.test_datasets, self.test_dataframes, _ = self.create_datasets(self.test_df, level_label_dicts=self.level_label_dicts, image_dict=self.test_image_dict, max_samples_per_class=self.config["max_samples_per_class"])
            
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
                "Species accuracy":ClasswiseWrapper(Accuracy(average="none", num_classes=num_classes, task="multiclass"), labels=taxon_level_labels),
                "Species precision":ClasswiseWrapper(Precision(average="none", num_classes=num_classes, task="multiclass"),labels=taxon_level_labels),
                })
                self.level_metrics[key] = level_metric
    
            self.classes = len(self.train_df.label.unique())
            for key, value in self.train_dataframes.items(): 
                classes = len(value.label.unique())
                base = base_model(classes=classes, years=self.years, config=self.config, name=key)
                self.models[key] = base   
            
    def dominant_class_model(self, df, image_dict=None):
        """A level 0 model splits out the dominant class and compares to all other samples"""
        level_label_dict = {value:key for key, value in enumerate(self.common_species)}
        if len(df.taxonID.unique()) < self.config["max_flat_species"]:
            raise ValueError("There are only {} species, choosing flat model".format(len(df.taxonID.unique())))

        if len(self.conifer_species) == 1:
            level_label_dict[self.conifer_species[0]] = len(level_label_dict)    
        elif len(self.conifer_species) > 1:
            level_label_dict["CONIFER"] = len(level_label_dict)  
        if len(self.broadleaf_species) == 1:
            level_label_dict[self.broadleaf_species[0]] = len(level_label_dict)        
        if len(self.broadleaf_species) > 1:
            level_label_dict["BROADLEAF"] = len(level_label_dict) 
        if not len(level_label_dict) > 1:
            raise ValueError("No dominant species or broadleaf or conifer split")
    
        # Select head and tail classes
        head_classes = df[df.taxonID.isin(self.common_species)]
        
        #Split tail classes into conifer and broadleaf
        tail_classes = df[~df.taxonID.isin(head_classes.taxonID)]
        if len(self.conifer_species) > 1:        
            tail_classes.loc[tail_classes.taxonID.isin(self.conifer_species),"taxonID"] = "CONIFER"
        if len(self.broadleaf_species) > 1:
            tail_classes.loc[tail_classes.taxonID.isin(self.broadleaf_species),"taxonID"] = "BROADLEAF"
        
        # Create labels
        level_0 = pd.concat([head_classes, tail_classes])                
        try:
            level_0["label"] = [level_label_dict[x] for x in level_0.taxonID]
        except:
            raise
        level_0_ds = TreeDataset(df=level_0, config=self.config, image_dict=image_dict)
        
        return level_0_ds, level_0, level_label_dict
    
    def conifer_model(self, df, image_dict=None):        
        level_label_dict = {value:key for key, value in enumerate(self.conifer_species)}
                    
        # Select head and tail classes
        level_1 = df[df.taxonID.isin(self.conifer_species)]
        
        # Create labels
        level_1["label"] = [level_label_dict[x] for x in level_1.taxonID]
        level_1_ds = TreeDataset(df=level_1, config=self.config, image_dict=image_dict)
        
        return level_1_ds, level_1, level_label_dict
    
    def broadleaf_model(self, df, image_dict=None):
        """Model for the broadleaf species"""
        
        if self.config["seperate_oak_model"]:
            oak = [x for x in self.broadleaf_species if x[0:2] == "QU"]   
            non_oak = [x for x in self.broadleaf_species if not x[0:2] == "QU"]        
            level_2 = df[df.taxonID.isin(self.broadleaf_species)]
            level_label_dict = {value:key for key, value in enumerate(non_oak)}
            
            if (len(oak) > 2) & (len(non_oak) > 1):
                oak = [x for x in self.broadleaf_species if x[0:2] == "QU"]
                level_label_dict["OAK"] = len(level_label_dict) 
                level_2.loc[level_2.taxonID.isin(oak),"taxonID"] = "OAK"
            else:
                for x in oak:
                    level_label_dict[x] = len(level_label_dict)
        else:
            level_2 = df[df.taxonID.isin(self.broadleaf_species)]
            level_label_dict = {value:key for key, value in enumerate(self.broadleaf_species)}            
            
        # Select head and tail classes
        level_2["label"] = [level_label_dict[x] for x in level_2.taxonID]
        level_2_ds = TreeDataset(df=level_2, config=self.config, image_dict=image_dict)
        
        return level_2_ds, level_2, level_label_dict
        
    def oak_model(self, df, image_dict=None): 
        oak = [x for x in self.broadleaf_species if x[0:2] == "QU"]        
        if not len(oak) > 2:
            raise ValueError("Not enough Oak Species")
        
        level_label_dict = {value:key for key, value in enumerate(oak)}
            
        # Select head and tail classes
        level_3 = df[df.taxonID.isin(oak)]
        level_3["label"] = [level_label_dict[x] for x in level_3.taxonID]
        level_3_ds = TreeDataset(df=level_3, config=self.config, image_dict=image_dict)
        
        return level_3_ds, level_3, level_label_dict
    
    def flat_model(self, df, image_dict=None):        
        level_label_dict = {value:key for key, value in enumerate(df.taxonID.unique())}
                            
        # Create labels
        df["label"] = [level_label_dict[x] for x in df.taxonID]
        ds = TreeDataset(df=df, config=self.config, image_dict=image_dict)
        
        return ds, df, level_label_dict
    
    def create_datasets(self, df, level_label_dicts=None, image_dict=None, max_samples_per_class=None):
        """Create a hierarchical set of dataloaders by splitting into taxonID groups
        train: whether to sample the training labels
        max_samples_per_class: limit the number of samples per class for balancing
        """
        datasets = {}
        dataframes = {}
        level_label_dicts = {}
        
        if max_samples_per_class:
            ids_to_keep = df.drop_duplicates(subset=["individual"]).groupby("taxonID").apply(lambda x: x.head(max_samples_per_class)).reset_index(drop=True)            
            df = df[df.individual.isin(ids_to_keep.individual)]
        try:
            level_ds, level_df, level_label_dict = self.dominant_class_model(df=df, image_dict=image_dict)
            level_label_dicts["dominant_class"] = level_label_dict
            dataframes["dominant_class"] = level_df
            datasets["dominant_class"] = level_ds            
        except:
            print("Not enough species to split into broadleaf, conifer and head class, running a flat model instead")
            traceback.print_exc()  
            level_ds, level_df, level_label_dict = self.flat_model(df=df, image_dict=image_dict)  
            level_label_dicts["flat"] = level_label_dict
            dataframes["flat"] = level_df
            datasets["flat"] = level_ds              
        
        if (len(self.conifer_species) > 1) and ("flat" not in dataframes.keys()):
            level_ds, level_df, level_label_dict = self.conifer_model(df, image_dict=image_dict)
            level_label_dicts["conifer"] = level_label_dict
            dataframes["conifer"] = level_df
            datasets["conifer"] = level_ds           
        
        if (len(self.broadleaf_species) > 1) and ("flat" not in dataframes.keys()):
            level_ds, level_df, level_label_dict = self.broadleaf_model(df, image_dict=image_dict)
            level_label_dicts["broadleaf"] = level_label_dict
            dataframes["broadleaf"] = level_df
            datasets["broadleaf"] = level_ds  
            
            if self.config["seperate_oak_model"]:
                try:
                    level_ds, level_df, level_label_dict = self.oak_model(df, image_dict=image_dict)
                    level_label_dicts["oak"] = level_label_dict
                    dataframes["oak"] = level_df
                    datasets["oak"] = level_ds           
                except:
                    print("Oak model failed")
                    traceback.print_exc()
                
        return datasets, dataframes, level_label_dicts
    
    def train_dataloader(self):        
        """Given the current hierarchical level, train a model and dataset"""
        imbalance = self.train_dataframes[self.current_level]
        ds = self.train_datasets[self.current_level]
        one_hot = torch.nn.functional.one_hot(torch.tensor(imbalance.groupby("individual", sort=False).apply(lambda x: x.head(1)).label.values))
        train_sampler = sampler.MultilabelBalancedRandomSampler(
            labels=one_hot, indices=range(len(imbalance.individual.unique())), class_choice="cycle")
        
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["batch_size"],
            sampler=train_sampler,
            num_workers=self.config["workers"]
        )
    
        return data_loader        

    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.test_datasets[self.current_level],
            batch_size=self.config["batch_size"],
            shuffle=False,
            num_workers=self.config["workers"]
        )
        
        return data_loader 
    
    def predict_dataloader(self, ds):
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["predict_batch_size"],
            shuffle=False,
            num_workers=self.config["workers"],
            collate_fn=utils.skip_none_collate
        )

        return data_loader
        
    def configure_optimizers(self):
        """Create a optimizer for each level"""
        optimizer = torch.optim.Adam(self.models[self.current_level].parameters(), lr=self.config["lr_{}".format(self.current_level)])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=4,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         eps=1e-08)
        
        return {'optimizer':optimizer, 'lr_scheduler':
                {"scheduler":scheduler, "monitor":'val_loss_{}'.format(self.current_level),
                 "frequency":self.config["validation_interval"],
                 "interval":"epoch"}}
        
    def training_step(self, batch, batch_idx):
        """Calculate train_df loss
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[self.current_level].forward(images)
        loss = F.cross_entropy(y_hat, y)    
        self.log("train_loss_{}".format(self.current_level),loss, on_epoch=True, on_step=False)

        return loss
    
    def validation_step(self, batch, batch_idx):
        """Calculate val loss and on_epoch metrics
        """
        individual, inputs, y = batch
        images = inputs["HSI"]  
        y_hat = self.models[self.current_level].forward(images)
        loss = F.cross_entropy(y_hat, y)   
        
        self.log("val_loss_{}".format(self.current_level),loss,add_dataloader_idx=False)
        try:
            self.models[self.current_level].metrics(y_hat, y)
        except:
            print("Validation failed with targets {}".format(y))
            
        self.log_dict(self.models[self.current_level].metrics, on_epoch=True, on_step=False)
        y_hat = F.softmax(y_hat, dim=1)
        
        self.level_metrics[self.current_level].update(y_hat, y)
 
        return {"individual":individual, "yhat":y_hat, "label":y}  
    
    def predict_step(self, batch, batch_idx):
        """Calculate predictions
        """
        if batch is None:
            return None
        
        individual, inputs = batch
        images = inputs["HSI"]  

        y_hats = []
        for level in self.models:   
            y_hat = self.models[level].forward(images)  
            if y_hat is None:
                raise ValueError("images of length {} with keys {} failed on predict step {}".format(len(images), images.keys(), images))          
            y_hat = F.softmax(y_hat, dim=1)
            y_hats.append(y_hat)
        
        return individual, y_hats

    def on_validation_epoch_end(self):
        class_metrics = self.level_metrics[self.current_level].compute()
        self.log_dict(class_metrics, on_epoch=True, on_step=False)
        self.level_metrics[self.current_level].reset()
        
    def gather_predictions(self, predict_df):
        """Post-process the predict method to create metrics"""
        individuals = []
        yhats = []
        levels = []
        
        for output in predict_df:
            if output is None:
                continue
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
        if results.empty:
            raise ValueError("No predictions made")
        
        for level, label_dict in self.label_to_taxonIDs.items():
            results["{}_taxa".format(level)] = results["{}_label".format(level)].apply(lambda x: label_dict[x])
        
        return results
    
    def ensemble(self, results):
        """Given a multi-level model, create a final output prediction and score"""
        ensemble_taxonID = []
        ensemble_label = []
        ensemble_score = []
                
        #For each level, select the predicted taxonID and retrieve the original label order
        for index, row in results.iterrows():            
            if "flat_taxa" in results.columns:
                if pd.isnull(row["flat_taxa"]):
                    ensemble_taxonID.append(np.nan)
                    ensemble_label.append(np.nan)
                    ensemble_score.append(np.nan) 
                    continue
                ensemble_taxonID.append(row["flat_taxa"])
                ensemble_label.append(self.species_label_dict[row["flat_taxa"]])
                ensemble_score.append(row["flat_score"])                   
            elif not row["dominant_class_taxa"] in ["CONIFER","BROADLEAF"]:
                if pd.isnull(row["dominant_class_taxa"]):
                    ensemble_taxonID.append(np.nan)
                    ensemble_label.append(np.nan)
                    ensemble_score.append(np.nan) 
                    continue
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
                ensemble_score.append(row["oak_score"])                
                
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
            task="multiclass",
            num_classes=len(self.species_label_dict)
        )
            
        ensemble_macro_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="macro",
            task="multiclass",
            num_classes=len(self.species_label_dict)
        )
        
        ensemble_precision = torchmetrics.functional.precision(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            num_classes=len(self.species_label_dict),
            task="multiclass"
        )
                        
        if experiment:
            experiment.log_metric("overall_macro", ensemble_macro_accuracy)
            experiment.log_metric("overall_micro", ensemble_accuracy) 
            experiment.log_metric("overal_precision", ensemble_precision) 
        
        #Species Accuracy
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="none",
            num_classes=len(self.species_label_dict),
            task="multiclass"
        )
            
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(ensemble_df.ens_label.values),
            target=torch.tensor(ensemble_df.label.values),
            average="none",
            num_classes=len(self.species_label_dict),
            task="multiclass"
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
                    task="multiclass",
                    num_classes=len(self.species_label_dict))
                                
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro) 
                
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro], "Macro Recall": [site_macro]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)        
        
        return ensemble_df
