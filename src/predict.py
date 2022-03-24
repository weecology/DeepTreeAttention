#Predict
import glob
import geopandas as gpd
import numpy as np
import os
import rasterio
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
from src.main import TreeModel
from src.models import dead
from src.utils import preprocess_image, ensemble
from src.CHM import postprocess_CHM
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data.dataloader import default_collate
import pandas as pd

def RGB_transform(augment):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(transforms.Resize([224,224]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)

class on_the_fly_dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       image_path: .tif file location
    """
    def __init__(self, crowns, image_path, data_type="HSI", config=None):
        self.config = config 
        self.crowns = crowns
        self.image_size = config["image_size"]
        self.data_type = data_type
        
        if data_type == "HSI":
            self.HSI_src = rasterio.open(image_path)
        elif data_type == "RGB":
            self.RGB_src = rasterio.open(image_path)
            self.transform = RGB_transform(augment=False)
        else:
            raise ValueError("data_type is {}, only HSI and RGB data types are currently allowed".format(data_type))
        
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        inputs = {}
        #Load crown and crop
        geom = self.crowns.iloc[index].geometry
        individual = self.crowns.iloc[index].individual
        left, bottom, right, top = geom.bounds
            
        #preprocess and batch
        if self.data_type =="HSI":
            crop = self.HSI_src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=self.HSI_src.transform))             
        
            if crop.size == 0:
                return individual, None
            
            image = preprocess_image(crop, channel_is_first=True)
            image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)

            inputs[self.data_type] = image
            
            return individual, inputs
        
        elif self.data_type=="RGB":
            #Expand RGB
            box = self.RGB_src.read(window=rasterio.windows.from_bounds(left-1, bottom-1, right+1, top+1, transform=self.RGB_src.transform))             
            #Channels last
            box = np.rollaxis(box,0,3)
            image = self.transform(box.astype(np.float32))
            image = image
            
            return image
        
def my_collate(batch):
    batch = [x for x in batch if x[1] is not None]
    
    return default_collate(batch)

def predict_tile(PATH, species_model_dir, config, dead_model_path=None):
    """Generate species prediction from a HSI tile"""
    #get rgb from HSI path
    HSI_basename = os.path.basename(PATH)
    if "hyperspectral" in HSI_basename:
        rgb_name = "{}.tif".format(HSI_basename.split("_hyperspectral")[0])    
    else:
        rgb_name = HSI_basename           
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    rgb_path = [x for x in rgb_pool if rgb_name in x][0]
    crowns = predict_crowns(rgb_path)
    crowns["tile"] = PATH
    
    #CHM filter
    if config["CHM_pool"]:
        CHM_pool = glob.glob(config["CHM_pool"], recursive=True)
        crowns = postprocess_CHM(crowns, CHM_pool)
        #Rename column
        filtered_crowns = crowns[crowns.CHM_height > 3]
    else:
        filtered_crowns = crowns
    
    #Load Alive/Dead model
    print(filtered_crowns.head())
    if filtered_crowns.empty:
        raise ValueError("No crowns left after CHM filter. {}".format(crowns.head(n=10)))
    
    if dead_model_path:
        dead_label, dead_score = predict_dead(
            crowns=filtered_crowns,
            dead_model_path=dead_model_path,
            rgb_tile=rgb_path,
            config=config)
        
        filtered_crowns["dead_label"] = dead_label
        filtered_crowns["dead_score"] = dead_score
    
    # Load species models
    model_paths = glob.glob(os.path.join(species_model_dir,"*.pl"))
    models = []
    for x in model_paths:
        m = TreeModel.load_from_checkpoint(x)        
        models.append(m)
    
    # Predict
    trees, features = predict_species(HSI_path=PATH, crowns=filtered_crowns, models=models, config=config)

    # Remove predictions for dead trees
    trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"pred_taxa_top1"] = "DEAD"
    trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"pred_label_top1"] = None
    trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"top1_score"] = None
    
    # Calculate crown area
    trees["crown_area"] = crowns.geometry.area
        
    return trees

def predict_crowns(PATH):
    """Predict a set of tree crowns from RGB data"""
    m = main.deepforest()
    if torch.cuda.is_available():
        m.config["gpus"] = 1
    m.use_release(check_release=False)
    boxes = m.predict_tile(PATH)
    r = rasterio.open(PATH)
    transform = r.transform     
    crs = r.crs
    gdf = annotations_to_shapefile(boxes, transform=transform, crs=crs)
    
    #Dummy variables for schema
    basename = os.path.splitext(os.path.basename(PATH))[0]
    individual = ["{}_{}".format(x, basename) for x in range(gdf.shape[0])]
    gdf["individual"] = individual
    gdf["plotID"] = None
    gdf["siteID"] = None #TODO
    gdf["box_id"] = None
    gdf["plotID"] = None
    gdf["taxonID"] = None
    gdf["RGB_tile"] = PATH
    
    return gdf

def predict_species(crowns, HSI_path, models, config):
    """Given a shapefile and HSI path, apply a model ensemble"""
    # Prepare data for ensemble prediction
    crowns["bbox_score"] = crowns["score"]    
    ds = on_the_fly_dataset(crowns=crowns, image_path=HSI_path, config=config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        collate_fn=my_collate
    )
    
    # Predict species for each year
    year_individuals = {} 
    year_results = []
    for year, m in enumerate(models):
        results, features = m.predict_dataloader(
            data_loader=data_loader,
            return_features=True
        )
        
        for index, row in enumerate(features):
            try:
                year_individuals[results.individual.iloc[index]].append(row)
            except:
                year_individuals[results.individual.iloc[index]] = [row]

        results["year"] = year
        year_results.append(results)

    # Ensemble and merge into original frame
    results = pd.concat(year_results)
    results = ensemble(results, year_individuals)
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score","RGB_tile"])]
    results = results.merge(crowns, on="individual")
    
    return results, features

def predict_dead(crowns, dead_model_path, rgb_tile, config):
    """Classify RGB crops as Alive/Dead"""
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    ds = on_the_fly_dataset(crowns=crowns, image_path=rgb_tile, config=config, data_type="RGB")
    label, score = dead.predict_dead_dataloader(dead_model, ds, config)
    
    return label, score


    
