#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import geopandas as gpd
import os
import numpy as np
import rasterio
from src.models import dead
from src.utils import preprocess_image, predictions_to_df, load_image
from src import patches
from src.CHM import postprocess_CHM
from src.generate import generate_crops
from torch.utils.data import Dataset
from torchvision import transforms
import torch

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

class predict_crops(Dataset):
    """A csv file with a path to image crop and label
    Args:
       df: pandas df image locations
       years: list of years to lookup in df
    """
    def __init__(self, crowns, years, config):
        #Load each tile into memory
        self.tiles = []
        self.crowns = crowns
        self.individuals = self.crowns.individual.unique()
        self.years = years
        self.config = config
        
    def __len__(self):
        #0th based index
        return len(self.individuals)
        
    def __getitem__(self, index):
        images = []        
        individual = self.individuals[index] 
        ind_annotations = self.crowns[self.crowns.individual == self.individuals[index]]
        for yr in self.years:
            yr_annotation = ind_annotations[ind_annotations.tile_year==str(yr)]
            if yr_annotation.empty:
                image = torch.zeros(self.config["bands"], self.config["image_size"], self.config["image_size"])
            else:
                image_path = os.path.join(self.config["prediction_crop_dir"], yr_annotation["image_path"].iloc[0])
                try:
                    image = np.load(image_path)
                except Exception as e:
                    print(e)
                    image = torch.zeros(self.config["bands"], self.config["image_size"], self.config["image_size"])                    
                image = preprocess_image(image=image, channel_is_first=True)
                image = transforms.functional.resize(image, size=(self.config["image_size"],self.config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)                
            images.append(image)
        inputs = {}
        inputs["HSI"] = images
        
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
        dead_label, dead_score = predict_dead(crowns=filtered_crowns, dead_model_path=dead_model_path, config=config)
        filtered_crowns["dead_label"] = dead_label
        filtered_crowns["dead_score"] = dead_score
    
    return filtered_crowns

def generate_prediction_crops(crowns, config, client=None, as_numpy=True):
    """Create prediction crops for model.predict"""
    
    crown_annotations = generate_crops(
        crowns,
        savedir=config["prediction_crop_dir"],
        sensor_glob=config["HSI_sensor_pool"],
        convert_h5=config["convert_h5"],   
        rgb_glob=config["rgb_sensor_pool"],
        HSI_tif_dir=config["HSI_tif_dir"],
        client=client,
        as_numpy=as_numpy
    )
    
    #Write file alongside
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score","RGB_tile"])]    
    crown_annotations = crown_annotations.merge(crowns)
    rgb_path = crown_annotations.RGB_tile.unique()[0]
    basename = os.path.splitext(os.path.basename(rgb_path))[0]         
    crown_annotations = gpd.GeoDataFrame(crown_annotations, geometry="geometry")    
    crown_annotations.to_file("{}/{}.shp".format(config["prediction_crop_dir"],basename))  
    
    return "{}/{}.shp".format(config["prediction_crop_dir"],basename)

def predict_tile(crown_annotations,m, trainer, config, savedir, filter_dead=False):
    """Predict a set of crown labels from a annotations.shp
    Args:
        crown_annotations: geodataframe from predict.generate_prediction_crops
        m: pytorch model to predict data
        config: config.yml
        savedir: directory to save tile predictions
        filter_dead: filter dead model
    """
    crown_annotations = gpd.read_file(crown_annotations) 
    trees = predict_species(crowns=crown_annotations, years=m.years, m=m, config=config, trainer=trainer)

    if trees is None:
        return None
    
    # Remove predictions for dead trees
    if filter_dead:
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ensembleTaxonID"] = "DEAD"
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_label"] = None
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_score"] = None
        
    # Calculate crown area
    trees["crown_area"] = crown_annotations.geometry.area
    trees["geometry"] = crown_annotations.geometry
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    
    print("{} trees predicted".format(trees.shape[0]))
    
    #Save .shp
    basename = os.path.splitext(os.path.basename(crown_annotations.RGB_tile.unique()[0]))[0]
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

def predict_species(crowns, years, m, trainer, config):
    """Compute hierarchical prediction without predicting unneeded levels"""
    # Level 0 PIPA v all
    ds = predict_crops(crowns=crowns, years=years, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    results = m.gather_predictions(predictions)
    crowns = results.merge(crowns, on="individual")
    ensemble_df = m.ensemble(results)
    crowns = crowns.loc[:,crowns.columns.isin(["individual","geometry","bbox_score","tile","CHM_height","dead_label","dead_score","RGB_tile"])]
    ensemble_df = ensemble_df.merge(crowns, on="individual")
        
    return ensemble_df

def predict_dead(crowns, dead_model_path, config):
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    ds = dead.utm_dataset(crowns=crowns, config=config)
    label, score = dead.predict_dead_dataloader(dead_model, ds, config)
    
    return label, score


