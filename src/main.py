#Lightning Data Module
import tempfile
import os

import geopandas as gpd
import glob as glob
from deepforest.main import deepforest
from descartes import PolygonPatch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from pytorch_lightning import LightningModule
import pandas as pd
from torch.nn import functional as F
import torch
from torchvision import transforms
import torchmetrics
import rasterio
from rasterio.plot import show
from shapely.geometry import Point, box
from src import data
from src import generate
from src import neon_paths
from src import patches
from src import utils
from . import __file__

class TreeModel(LightningModule):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self, model, classes, label_dict, loss_weight=None, config=None, *args, **kwargs):
        super().__init__()
        self.ROOT = os.path.dirname(os.path.dirname(__file__))    
        self.tmpdir = tempfile.gettempdir()
        if config is None:
            self.config = utils.read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config

        self.classes = classes
        self.label_to_index = label_dict
        self.index_to_label = {}
        for x in label_dict:
            self.index_to_label[label_dict[x]] = x 

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
                
        #Weighted loss
        if isinstance(loss_weight, type(None)):
            loss_weight = torch.ones((classes))         
            
        if torch.cuda.is_available():
            self.loss_weight = torch.tensor(loss_weight, device="cuda", dtype=torch.float)
        else:
            self.loss_weight = torch.tensor(loss_weight, dtype=torch.float)
            
        self.save_hyperparameters(ignore=["loss_weight"])
        

    def training_step(self, batch, batch_idx):
        """Calculate train loss
        """
        individual, inputs, y = batch
        images = inputs["HSI"]
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat[-1], y, weight=self.loss_weight)    

        return loss


    def validation_step(self, batch, batch_idx):
        """Calculate val loss
        """
        individual, inputs, y = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat[-1], y, weight=self.loss_weight)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        
        return loss

    def on_validation_epoch_end(self):
        results = self.predict_dataloader(self.trainer.datamodule.val_dataloader())
        
        final_micro = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values),
            average="micro")
        
        final_macro = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values),
            average="macro",
            num_classes=self.classes)
        
        self.log("Epoch Micro Accuracy", final_micro)
        self.log("Epoch Macro Accuracy", final_macro)
        
        # Log results by species
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values), 
            average="none", 
            num_classes=self.classes
        )
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values),
            average="none",
            num_classes=self.classes
        )
        species_table = pd.DataFrame(
            {"taxonID":self.label_to_index.keys(),
             "accuracy":taxon_accuracy,
             "precision":taxon_precision
             })
        
        for key, value in species_table.set_index("taxonID").accuracy.to_dict().items():
            self.log("Epoch_{}_accuracy".format(key), value)
            
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.75,
                                                         patience=8,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=0.0000001,
                                                         eps=1e-08)

        return {'optimizer':optimizer, 'lr_scheduler': scheduler, "monitor":'val_loss'}
    

    def predict_image(self, img_path, return_numeric = False):
        """Given an image path, load image and predict"""
        self.model.eval()        
        image = data.load_image(img_path, image_size=self.config["image_size"])
        batch = torch.unsqueeze(image, dim=0)
        with torch.no_grad():
            y = self.model(batch)  
        index = np.argmax(y[-1], 1).numpy()[0]
        
        if return_numeric:
            return index
        else:
            return self.index_to_label[index]
                
    def predict_xy(self, coordinates, fixed_box=True):
        """Given an x,y location, find sensor data and predict tree crown class. If no predicted crown within 5m an error will be raised (fixed_box=False) or a 1m fixed box will created (fixed_box=True)
        Args:
            coordinates (tuple): x,y tuple in utm coordinates
            fixed_box (False): If no DeepForest tree is predicted within 5m of centroid, create a 1m fixed box. If false, raise ValueError
        Returns:
            label: species taxa label
        """
        gdf = gpd.GeoDataFrame(geometry=[Point(coordinates[0],coordinates[1])])
        img_pool = glob.glob(self.config["rgb_sensor_pool"], recursive=True)
        rgb_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)

        # DeepForest model to predict crowns
        deepforest_model = deepforest()
        deepforest_model.use_release(check_release=False)
        boxes = generate.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_path, bounds=gdf.total_bounds, expand=40)
        boxes['geometry'] = boxes.apply(lambda x: box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)

        if boxes.shape[0] > 1:
            centroid_distances = boxes.centroid.distance(Point(coordinates[0], coordinates[1])).sort_values()
            centroid_distances = centroid_distances[centroid_distances<5]
            boxes = boxes[boxes.index == centroid_distances.index[0]]
        if boxes.empty:
            if fixed_box:
                boxes = generate.create_boxes(boxes)
            else:
                raise ValueError("No predicted tree centroid within 5 m of point {}, to ignore this error and specify fixed_box=True".format(coordinates))
            
        # Create pixel crops
        img_pool = glob.glob(self.config["HSI_sensor_pool"], recursive=True)        
        sensor_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)        
        crop = patches.crop(
            bounds=boxes["geometry"].values[0].bounds,
            sensor_path=sensor_path,
        )

        # Preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(self.config["image_size"], self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        image = torch.unsqueeze(image, dim = 0)

        # Classify pixel crops
        self.model.eval() 
        with torch.no_grad():
            class_probs = self.model(image)
            class_probs = F.softmax(class_probs[-1])
        class_probs = class_probs.numpy()
        index = np.argmax(class_probs)
        label = self.index_to_label[index]
        
        #Average score for selected label
        score = class_probs[:,index]
        
        return label, score
    
    def predict_crown(self, geom, sensor_path):
        """Given a geometry object of a tree crown, predict label
        Args:
            geom: a shapely geometry object, for example from a geodataframe (gdf) -> gdf.geometry[0]
            sensor_path: path to sensor data to predict
        Returns:
            label: taxonID
            average_score: average pixel confidence for of the prediction class
        """
        crop = patches.crop(
            bounds=geom.bounds,
            sensor_path=sensor_path
        )
     
        #preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(self.config["image_size"], self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        image = torch.unsqueeze(image, dim = 0)
        
        #Classify pixel crops
        self.model.eval() 
        with torch.no_grad():
            class_probs = self.model(image) 
            class_probs = F.softmax(class_probs[-1], 1)
        class_probs = class_probs.detach().numpy()
        index = np.argmax(class_probs)
        label = self.index_to_label[index]
        
        #Average score for selected label
        score = class_probs[:,index]
        
        return label, score
    
    def predict(self,inputs):
        """Given a input dictionary, construct args for prediction"""
        if "cuda" == self.device.type:
            images = inputs["HSI"]
            images = images.cuda()
            pred = self.model(images)
            #Last spectral block
            pred = pred[-1]
            pred = pred.cpu()
        else:
            images = inputs["HSI"]
            pred = self.model(images)
            #Last spectral block            
            pred = pred[-1]
        
        return pred
    
    def predict_dataloader(self, data_loader, return_features=False, train=True):
        """Given a file with paths to image crops, create crown predictions 
        The format of image_path inform the crown membership, the files should be named crownid_counter.png where crownid is a
        unique identifier for each crown and counter is 0..n pixel crops that belong to that crown.
        
        Args: 
            csv_file: path to csv file
            data_loader: data.TreeData loader
            plot_n_individuals: if experiment, how many plots to create
            return_features (False): If true, return a samples x classes matrix of softmax features
        Returns:
            results: if return_features == False, pandas dataframe with columns crown and species label
            features: if return_features == True, samples x classes matrix of softmax features
        """
        self.model.eval()
        predictions = []
        labels = []
        individuals = []
        for batch in data_loader:
            if train:
                individual, inputs, targets = batch
            else:
                individual, inputs = batch
            with torch.no_grad():
                pred = self.predict(inputs)
                pred = F.softmax(pred, dim=1)
            predictions.append(pred)
            individuals.append(individual)
            if train:
                labels.append(targets)                

        individuals = np.concatenate(individuals)        
        predictions = np.concatenate(predictions) 
        if train:
            labels = np.concatenate(labels)
        
        # Concat batches
        predictions_top1 = np.argmax(predictions, 1)    
        predictions_top2 = pd.DataFrame(predictions).apply(lambda x: np.argsort(x.values)[-2], axis=1)
        top1_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[0], axis=1)
        top2_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[1], axis=1)
        
        # Construct a df of predictions
        df = pd.DataFrame({
            "pred_label_top1":predictions_top1,
            "pred_label_top2":predictions_top2,
            "top1_score":top1_score,
            "top2_score":top2_score,
            "individual":individuals
        })
        df["pred_taxa_top1"] = df["pred_label_top1"].apply(lambda x: self.index_to_label[x]) 
        df["pred_taxa_top2"] = df["pred_label_top2"].apply(lambda x: self.index_to_label[x])        
        if train:
            df["label"] = labels
            df["true_taxa"] = df["label"].apply(lambda x: self.index_to_label[x])            
    
        if return_features:            
            return df, predictions        
        else:
            return df
    
    def evaluate_crowns(self, data_loader, crowns, experiment=None):
        """Crown level measure of accuracy
        Args:
            data_loader: TreeData dataset
            experiment: optional comet experiment
            points: the canopy_points.shp from the data_module
        Returns:
            df: results dataframe
            metric_dict: metric -> value
        """
        results, features = self.predict_dataloader(
            data_loader=data_loader,
            return_features=True
        )
        
        # Read in crowns data
        results = results.merge(crowns.drop(columns="label"), on="individual")
        results = gpd.GeoDataFrame(results, geometry="geometry")

        # Log results by species
        taxon_accuracy = torchmetrics.functional.accuracy(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values), 
            average="none", 
            num_classes=self.classes
        )
        taxon_precision = torchmetrics.functional.precision(
            preds=torch.tensor(results.pred_label_top1.values),
            target=torch.tensor(results.label.values),
            average="none",
            num_classes=self.classes
        )
        species_table = pd.DataFrame(
            {"taxonID":self.label_to_index.keys(),
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
                
                site_micro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.pred_label_top1.values),
                    target=torch.tensor(group.label.values),
                    average="micro")
                
                site_macro = torchmetrics.functional.accuracy(
                    preds=torch.tensor(group.pred_label_top1.values),
                    target=torch.tensor(group.label.values),
                    average="macro",
                    num_classes=self.classes)
                
                experiment.log_metric("{}_macro".format(name), site_macro)
                experiment.log_metric("{}_micro".format(name), site_micro) 
                
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro.numpy()], "Macro Recall": [site_macro.numpy()]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)
        
        return results
            