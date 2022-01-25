#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import geopandas as gpd
import rasterio
from src.main import TreeModel
from src import data 
from torch.utils.data import Dataset
import os
import numpy as np
from torchvision import transforms
from torch.nn import functional as F
import torch
from torch.utils.data.dataloader import default_collate

class on_the_fly_Dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       image_path: .tif file location
    """
    def __init__(self, crowns, image_path, config=None):
        self.config = config 
        self.crowns = crowns
        self.image_size = config["image_size"]
        self.src = rasterio.open(image_path)
        
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        geom = self.crowns.iloc[index].geometry
        individual = self.crowns.iloc[index].individual
        left, bottom, right, top = geom.bounds
        crop = self.src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=self.src.transform)) 
        
        if crop.size == 0:
            return individual, None
            
        #preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        
        inputs = {}
        inputs["HSI"] = image
    
        return individual, inputs
        

def my_collate(batch):
    batch = [x for x in batch if x[1] is not None]
    return default_collate(batch)
    
def predict_tile(PATH, model_path, config):
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
    
    #Load species model
    m = TreeModel.load_from_checkpoint(model_path)
    trees, features = predict_species(HSI_path=PATH, crowns=crowns, m=m, config=config)
    
    #Spatial smooth
    trees = smooth(trees=trees, features=features, size=config["neighbor_buffer_size"], alpha=config["neighborhood_strength"])
    trees["spatial_taxonID"] = trees["spatial_label"]
    trees["spatial_taxonID"] = trees["spatial_label"].apply(lambda x: m.index_to_label[x]) 
    
    return trees

def predict_crowns(PATH):
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
    
    return gdf

def predict_species(crowns, HSI_path, m, config):
    ds = on_the_fly_Dataset(crowns, HSI_path, config)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["predict_batch_size"],
        shuffle=False,
        num_workers=config["workers"],
        collate_fn=my_collate
    )
    df, features = m.predict_dataloader(data_loader, train=False, return_features=True)
    crowns["bbox_score"] = crowns["score"]
    
    #If CHM exists
    try:
        df = df.merge(crowns[["individual","geometry","bbox_score","tile","CHM_height"]], on="individual")
    except:
        df = df.merge(crowns[["individual","geometry","bbox_score","tile"]], on="individual")
    
    return df, features

def smooth(trees, features, size, alpha):
    """Given the results dataframe and feature labels, spatially smooth based on alpha value"""
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    sindex = trees.sindex
    tree_buffer = trees.buffer(size)
    smoothed_features = []
    for index, geom in enumerate(tree_buffer):
        intersects = sindex.query(geom)
        focal_feature = features[index,]
        neighbor_features = np.mean(features[intersects,], axis=0)
        smoothed_feature = focal_feature + alpha * neighbor_features
        smoothed_features.append(smoothed_feature)
    smoothed_features = np.vstack(smoothed_features)
    spatial_label = np.argmax(smoothed_features, axis=1)
    spatial_score = np.max(smoothed_features, axis=1)
    trees["spatial_label"] = spatial_label
    trees["spatial_score"] = spatial_score
    
    return trees
    
