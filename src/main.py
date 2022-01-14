#Lightning Data Module
from . import __file__
import geopandas as gpd
import glob as glob
from deepforest.main import deepforest
from descartes import PolygonPatch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from pytorch_lightning import LightningModule
import os
import pandas as pd
from torch.nn import functional as F
from torch import optim
import torch
from torchvision import transforms
import torchmetrics
import tempfile
import rasterio
from rasterio.plot import show
from src import data
from src import generate
from src import neon_paths
from src import patches
from src import spatial
from shapely.geometry import Point, box

class TreeModel(LightningModule):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self,model, classes, label_dict, config=None, *args, **kwargs):
        super().__init__()
    
        self.ROOT = os.path.dirname(os.path.dirname(__file__))    
        self.tmpdir = tempfile.gettempdir()
        if config is None:
            self.config = data.read_config("{}/config.yml".format(self.ROOT))   
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
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall,"Macro Accuracy":macro_recall,"Top {} Accuracy".format(self.config["top_k"]): top_k_recall})
        
        self.save_hyperparameters()
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]        
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        softmax_prob = F.softmax(y_hat, dim =1)
        output = self.metrics(softmax_prob, y) 
        self.log_dict(output)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=0.000001,
                                                         eps=1e-08)
                                                                 
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_loss'}
    
    def predict_image(self, img_path, return_numeric = False):
        """Given an image path, load image and predict"""
        self.model.eval()        
        image = data.load_image(img_path, image_size=self.config["image_size"])
        batch = torch.unsqueeze(image, dim=0)
        with torch.no_grad():
            y = self.model(batch)  
        index = np.argmax(y, 1).numpy()[0]
        
        if return_numeric:
            return index
        else:
            return self.index_to_label[index]
                
    def predict_xy(self, coordinates, fixed_box=True):
        #TODO update for metadata model
        """Given an x,y location, find sensor data and predict tree crown class. If no predicted crown within 5m an error will be raised (fixed_box=False) or a 1m fixed box will created (fixed_box=True)
        Args:
            coordinates (tuple): x,y tuple in utm coordinates
            fixed_box (False): If no DeepForest tree is predicted within 5m of centroid, create a 1m fixed box. If false, raise ValueError
        Returns:
            label: species taxa label
        """
        #Predict crown
        gdf = gpd.GeoDataFrame(geometry=[Point(coordinates[0],coordinates[1])])
        img_pool = glob.glob(self.config["rgb_sensor_pool"], recursive=True)
        rgb_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)
        
        #DeepForest model to predict crowns
        deepforest_model = deepforest()
        deepforest_model.use_release(check_release=False)
        boxes = generate.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_path, bounds=gdf.total_bounds, expand=40)
        boxes['geometry'] = boxes.apply(lambda x: box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
        
        if boxes.shape[0] > 1:
            centroid_distances = boxes.centroid.distance(Point(coordinates[0],coordinates[1])).sort_values()
            centroid_distances = centroid_distances[centroid_distances<5]
            boxes = boxes[boxes.index == centroid_distances.index[0]]
        if boxes.empty:
            if fixed_box:
                boxes = generate.create_boxes(boxes)
            else:
                raise ValueError("No predicted tree centroid within 5 m of point {}, to ignore this error and specify fixed_box=True".format(coordinates))
            
        #Create pixel crops
        img_pool = glob.glob(self.config["HSI_sensor_pool"], recursive=True)        
        sensor_path = neon_paths.find_sensor_path(lookup_pool=img_pool, bounds=gdf.total_bounds)        
        crop = patches.crop(
            bounds=boxes["geometry"].values[0].bounds,
            sensor_path=sensor_path,
        )
     
        #preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        image = torch.unsqueeze(image, dim = 0)
        
        #Classify pixel crops
        self.model.eval() 
        with torch.no_grad():
            class_probs = self.model(image)
            class_probs = F.softmax(class_probs)
        class_probs = class_probs.numpy()
        index = np.argmax(class_probs)
        label = self.index_to_label[index]
        
        #Average score for selected label
        score = class_probs[:,index]
        
        return label, score
    
    def predict_crown(self, geom, sensor_path):
        #TODO UPDATE for metadata model
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
        image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        image = torch.unsqueeze(image, dim = 0)
        
        #Classify pixel crops
        self.model.eval() 
        with torch.no_grad():
            class_probs = self.model(image) 
            class_probs = F.softmax(class_probs, 1)
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
            pred = pred.cpu()
        else:
            images = inputs["HSI"]
            pred = self.model(images)
            
        return pred
    
    def predict_dataloader(self, data_loader, plot_n_individuals=1, return_features=False, experiment=None):
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
            individual, inputs, targets = batch
            with torch.no_grad():
                pred = self.predict(inputs)
                pred = F.softmax(pred, dim=1)
            predictions.append(pred)
            labels.append(targets)
            individuals.append(individual)

        individuals = np.concatenate(individuals)        
        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions) 
        
        if return_features:            
            return predictions
        
        predictions_top1 = np.argmax(predictions, 1)    
        predictions_top2 = pd.DataFrame(predictions).apply(lambda x: np.argsort(x.values)[-2], axis=1)
        top1_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[0], axis=1)
        top2_score = pd.DataFrame(predictions).apply(lambda x: x.sort_values(ascending=False).values[1], axis=1)
        
        #Construct a df of predictions
        df = pd.DataFrame({"pred_label_top1":predictions_top1,"pred_label_top2":predictions_top2,"top1_score":top1_score,"top2_score":top2_score,"label":labels,"individual":individuals})
        df["pred_taxa_top1"] = df["pred_label_top1"].apply(lambda x: self.index_to_label[x]) 
        df["pred_taxa_top2"] = df["pred_label_top2"].apply(lambda x: self.index_to_label[x])        
        df["true_taxa"] = df["label"].apply(lambda x: self.index_to_label[x])
 
        if experiment:
            #load image pool and crown predicrions
            rgb_pool = glob.glob(self.config["rgb_sensor_pool"], recursive=True)
            test_points = gpd.read_file("{}/data/processed/canopy_points.shp".format(self.ROOT))   
            test_crowns = gpd.read_file("{}/data/processed/crowns.shp".format(self.ROOT))   
            
            plt.ion()
            for index, row in df.sample(n=plot_n_individuals).iterrows():
                fig = plt.figure(0)
                ax = fig.add_subplot(1, 1, 1)                
                individual = row["individual"]
                geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
                left, bottom, right, top = geom.bounds
                
                #Find image
                img_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=geom.bounds)
                src = rasterio.open(img_path)
                img = src.read(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
                img_transform = src.window_transform(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
                
                #Plot crown
                patches = [PolygonPatch(geom, edgecolor='red', facecolor='none')]
                show(img, ax=ax, transform=img_transform)                
                ax.add_collection(PatchCollection(patches, match_original=True))
                
                #Plot field coordinate
                stem = test_points[test_points.individual == individual]
                stem.plot(ax=ax)
                
                plt.savefig("{}/{}.png".format(self.tmpdir, row["individual"]))
                experiment.log_image("{}/{}.png".format(self.tmpdir, row["individual"]), name = "crown: {}, True: {}, Predicted {}".format(row["individual"], row.true_taxa,row.pred_taxa_top1))
                src.close()
                plt.close("all")
            plt.ioff()
            
        return df
    
    def evaluate_crowns(self, data_loader, experiment=None):
        """Crown level measure of accuracy
        Args:
            data_loader: TreeData dataset
            experiment: optional comet experiment
        Returns:
            df: results dataframe
            metric_dict: metric -> value
        """
        results = self.predict_dataloader(data_loader=data_loader, plot_n_individuals=self.config["plot_n_individuals"], experiment=experiment)
        features = self.predict_dataloader(data_loader=data_loader, return_features=True)
        
        #read in crowns data
        crowns = gpd.read_file("{}/data/processed/crowns.shp".format(self.ROOT))   
        results = results.merge(crowns.drop(columns="label"), on="individual")
        results = gpd.GeoDataFrame(results, geometry="geometry")
        HSI_pool = glob.glob(self.config["HSI_tif_dir"] +"*.tif")
        neighbors = spatial.spatial_neighbors(
            results,
            buffer=self.config["neighbor_buffer_size"],
            model = self,data_dir = "data/",
            image_size=self.config["image_size"],
            HSI_pool=HSI_pool)        
        
        #Spatial function
        labels, scores = spatial.spatial_smooth(neighbors, features, alpha=self.config["neighborhood_strength"])
        results["spatial_pred_label"] = labels
        results["spatial_score"] = scores
        
        spatial_micro = torchmetrics.functional.accuracy(preds=torch.tensor(results.spatial_pred_label.values),target=torch.tensor(results.label.values), average="micro")
        spatial_macro = torchmetrics.functional.accuracy(preds=torch.tensor(results.spatial_pred_label.values),target=torch.tensor(results.label.values), average="macro", num_classes=self.classes)
        if experiment:
            experiment.log_metric("spatial_micro",spatial_micro)
            experiment.log_metric("spatial_macro",spatial_macro)
            
        #Log result by site
        if experiment:
            site_data_frame =[]
            for name, group in results.groupby("siteID"):
                site_micro = torchmetrics.functional.accuracy(preds=torch.tensor(group.pred_label_top1.values),target=torch.tensor(group.label.values), average="micro")
                site_macro = torchmetrics.functional.accuracy(preds=torch.tensor(group.pred_label_top1.values),target=torch.tensor(group.label.values), average="macro", num_classes=self.classes)
                row = pd.DataFrame({"Site":[name], "Micro Recall": [site_micro.numpy()], "Macro Recall": [site_macro.numpy()]})
                site_data_frame.append(row)
            site_data_frame = pd.concat(site_data_frame)
            experiment.log_table("site_results.csv", site_data_frame)
        
        return results
            