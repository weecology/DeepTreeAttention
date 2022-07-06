#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import geopandas as gpd
import numpy as np
import os
import rasterio
import pandas as pd
from src.models import dead
from src import neon_paths
from src.utils import preprocess_image
from src import patches
from src.CHM import postprocess_CHM
from src.models import multi_stage
from torch.utils.data import Dataset
from torchvision import transforms
import torch
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

        else:
            raise ValueError("data_type is {}, only HSI and RGB data types are currently allowed".format(data_type))
        
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        inputs = {}
        #Load crown and crop
        self.RGB_src = rasterio.open(image_path)
        self.transform = RGB_transform(augment=False)
        
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
        

class predict_dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       image_paths: list of image_paths
    """
    def __init__(self, crowns, image_paths, config=None):
        #Load each tile into memory
        self.tiles = []
        self.crowns = crowns
        self.config = config
        
        for x in image_paths:
            #src = rasterio.open(x)            
            self.tiles.append(x)
    
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        bounds = self.crowns.geometry.iloc[index].bounds
        individual = self.crowns.individual.iloc[index]
        images = []
        for x in self.tiles:
            if x is None:
                image = torch.zeros(self.config["bands"], self.config["image_size"], self.config["image_size"])      
            else:
                crop = patches.crop(bounds, sensor_path=x)
                if crop is None:
                    image = torch.zeros(self.config["bands"], self.config["image_size"], self.config["image_size"])      
                else:
                    image = preprocess_image(crop, channel_is_first=True)
                    image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
            images.append(image)
        
        inputs = {}
        inputs["HSI"] = images
        
        if all([x.sum() == 0 for x in images]):
            return individual, None
        
        return individual, inputs
        
def find_crowns(rgb_path, config, dead_model_path=None):
    crowns = predict_crowns(rgb_path)
    if crowns is None:
        return None
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
        dead_config = config.copy()
        dead_config["workers"] = 0
        dead_label, dead_score = predict_dead(crowns=filtered_crowns, dead_model_path=dead_model_path, rgb_tile=rgb_path, config=dead_config)
        filtered_crowns["dead_label"] = dead_label
        filtered_crowns["dead_score"] = dead_score
    
    return filtered_crowns

def predict_tile(crowns, species_model_path, config, savedir, img_pool, filter_dead=False):        
    # Load species model
    m = multi_stage.MultiStage.load_from_checkpoint(species_model_path, config=config)
    year_paths = []
    for yr in m.years:
        geo_index =  neon_paths.bounds_to_geoindex(crowns.geometry.total_bounds)
        image_paths = neon_paths.find_sensor_path(lookup_pool=img_pool, geo_index=geo_index, all_years=True)
        try:
            year_path = [x for x in image_paths if "_{}.tif".format(yr) in x][0]
        except:
            year_path = None
        year_paths.append(year_path)
    
    if not len(year_paths) == len(m.years):
        raise ValueError("images paths {} does not match list of year models {}".format(image_paths, m.years))
    
    trees = predict_species(crowns=crowns, image_paths=year_paths, m=m, config=config)

    # Remove predictions for dead trees
    if filter_dead:
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ensembleTaxonID"] = "DEAD"
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_label"] = None
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_score"] = None
        
    # Calculate crown area
    trees["crown_area"] = crowns.geometry.area
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    basename = os.path.splitext(os.path.basename(crowns.RGB_tile.unique()[0]))[0]
    trees.to_file(os.path.join(savedir, "{}.shp".format(basename)))
    
    return trees

def predict_crowns(PATH):
    """Predict a set of tree crowns from RGB data"""
    m = main.deepforest()
    if torch.cuda.is_available():
        print("CUDA detected")
        m.config["gpus"] = 1
    m.use_release(check_release=False)
    boxes = m.predict_tile(PATH)
    if boxes is None:
        return None
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

def predict_species(crowns, image_paths, m, config):
    """Compute hierarchical prediction without predicting unneeded levels"""
    
    #Recursive predict to avoid prediction levels that will be later ignored.
    trainer = Trainer(gpus=config["gpus"], checkpoint_callback=False, logger=False, enable_checkpointing=False)
    
    # Level 0 PIPA v all
    ds = predict_dataset(crowns=crowns, image_paths=image_paths, config=config)
    m.current_level = 0
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=[ds]))    
    results = m.gather_predictions([predictions])
    print("{} crowns for level 0".format(len(ds)))
    
    # Level 1 Needleleaf v Broadleaf
    remaining_crowns = results[~(results["pred_taxa_top1_level_0"]=="PIPA2")].individual
    if len(remaining_crowns) > 0:
        m.current_level = 1        
        print("{} crowns for level 1".format(len(remaining_crowns)))
        level1_crowns = crowns[crowns.individual.isin(remaining_crowns)]
        ds = predict_dataset(crowns=level1_crowns, image_paths=image_paths, config=config)    
        predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=[ds]))
        results_1 = m.format_level(predictions, index=1, label_to_taxonIDs=m.label_to_taxonIDs[1])
        results = results.merge(results_1,on="individual", how="outer")

    # Level 2 Within Broadleaf                
    try:
        remaining_crowns = results_1[results_1["pred_taxa_top1_level_1"]=="BROADLEAF"].individual
    except:
        remaining_crowns = []
        
    if len(remaining_crowns) > 0: 
        m.current_level = 2        
        print("{} crowns for level 2".format(len(remaining_crowns)))        
        level2_crowns = crowns[crowns.individual.isin(remaining_crowns)]
        ds = predict_dataset(crowns=level2_crowns, image_paths=image_paths, config=config)    
        predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=[ds]))
        results_2 = m.format_level(predictions, index=2, label_to_taxonIDs=m.label_to_taxonIDs[2])
        results = results.merge(results_2,on="individual", how="outer")
        
    # Level 3 Within CONFIFER    
    try:
        remaining_crowns = results_1[results_1["pred_taxa_top1_level_1"]=="CONIFER"].individual
    except:
        remaining_crowns = []
        
    if len(remaining_crowns) > 0:
        m.current_level = 3        
        print("{} crowns for level 3".format(len(remaining_crowns)))                
        level3_crowns = crowns[crowns.individual.isin(remaining_crowns)]
        ds = predict_dataset(crowns=level3_crowns, image_paths=image_paths, config=config)    
        predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=[ds]))
        results_3 = m.format_level(predictions, index=3, label_to_taxonIDs=m.label_to_taxonIDs[3])
        results = results.merge(results_3,on="individual", how="outer")

    # Level 4 Within OAK
    try:
        remaining_crowns = results_2[results_2["pred_taxa_top1_level_2"]=="OAK"].individual
    except:
        remaining_crowns = []
        
    if len(remaining_crowns) > 0:
        m.current_level = 4        
        print("{} crowns for level 4".format(len(remaining_crowns)))                
        level4_crowns = crowns[crowns.individual.isin(remaining_crowns)]
        ds = predict_dataset(crowns=level4_crowns, image_paths=image_paths, config=config)    
        predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds_list=[ds]))
        results_4 = m.format_level(predictions, index=4, label_to_taxonIDs=m.label_to_taxonIDs[4])
        results = results.merge(results_4,on="individual", how="outer")

    crowns = results.merge(crowns, on="individual")
    ensemble_df = m.ensemble(results)
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score","RGB_tile"])]
    ensemble_df = ensemble_df.merge(crowns, on="individual")
    
    return ensemble_df

def predict_dead(crowns, dead_model_path, rgb_tile, config):
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    ds = on_the_fly_dataset(crowns=crowns, image_path=rgb_tile, config=config, data_type="RGB")
    label, score = dead.predict_dead_dataloader(dead_model, ds, config)
    
    return label, score


