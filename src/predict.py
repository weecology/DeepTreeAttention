#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import geopandas as gpd
import numpy as np
import os
import rasterio
import re
from src.main import TreeModel
from src.models import dead
from src import data
from src.utils import preprocess_image
from src.CHM import postprocess_CHM
from src.models import multi_stage
from src import generate
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torch.utils.data.dataloader import default_collate
from pytorch_lightning import Trainer
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

def generate_crops(crowns, config, dead_model_path=None):
    #Generate crops
    annotations = generate.generate_crops(
        crowns,
        savedir=config["crop_dir"],
        sensor_glob=config["HSI_sensor_pool"],
        convert_h5=config["convert_h5"],   
        rgb_glob=config["rgb_sensor_pool"],
        HSI_tif_dir=config["HSI_tif_dir"],
        replace=config["replace"]
    )
    
    return annotations

def find_crowns(rgb_path, config):
    crowns = predict_crowns(rgb_path)
    crowns["tile"] = rgb_path
    
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
        dead_label, dead_score = predict_dead(crowns=filtered_crowns, dead_model_path=dead_model_path, rgb_tile=rgb_path, config=config)
        filtered_crowns["dead_label"] = dead_label
        filtered_crowns["dead_score"] = dead_score
    
    return filtered_crowns

def predict_tile(annotations, species_model_path, config, savedir, RGB_year="2019", filter_dead=False):        
    # Load species model
    m = multi_stage.MultiStage.load_from_checkpoint(species_model_path)    
    trees = predict_species(annotations=annotations, m=m, config=config)

    # Remove predictions for dead trees
    if filter_dead:
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"pred_taxa_top1"] = "DEAD"
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"pred_label_top1"] = None
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"top1_score"] = None
        
    # Calculate crown area
    trees["crown_area"] = crowns.geometry.area
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    trees.to_file(os.path.join(savedir, "{}.shp".format(annotations.RGB_tile.unique()[0])))
    
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

def predict_species(annotations, m, config):
    predict_datasets = []
    for level in range(m.levels):
        ds = data.TreeDataset(df=annotations, train=False, config=config)
        predict_datasets.append(ds)

    trainer = Trainer(gpus=config["gpus"], checkpoint_callback=False)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=predict_datasets))
    results = m.gather_predictions(predictions)
    results["individualID"] = results["individual"]
    results = results.merge(crowns, on="individual")
    ensemble_df = m.ensemble(results)
    
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score","RGB_tile"])]
    crowns["individualID"] = crowns["individual"]
    ensemble_df = ensemble_df.merge(crowns, on="individualID")
    
    return ensemble_df

def predict_dead(crowns, dead_model_path, rgb_tile, config):
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    ds = on_the_fly_dataset(crowns=crowns, image_path=rgb_tile, config=config, data_type="RGB")
    label, score = dead.predict_dead_dataloader(dead_model, ds, config)
    
    return label, score


