#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import os
import geopandas as gpd
import rasterio
import numpy as np
from torchvision import transforms
import torch
from pytorch_lightning import Trainer

from src.models import dead
from src.CHM import postprocess_CHM
from src.generate import generate_crops
from src.data import TreeDataset

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
    
def find_crowns(rgb_path, config, dead_model_path=None):
    crowns = predict_crowns(rgb_path, config)
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
    rgb_path = crown_annotations.RGB_tile.unique()[0]
    basename = os.path.splitext(os.path.basename(rgb_path))[0]         
    crown_annotations = gpd.GeoDataFrame(crown_annotations, geometry="geometry")    
    crown_annotations = crown_annotations.merge(crowns[["individual","dead_label","dead_score"]])
    
    crown_annotations.to_file("{}/{}.shp".format(config["prediction_crop_dir"],basename))  
    
    return "{}/{}.shp".format(config["prediction_crop_dir"], basename)

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
    trees = predict_species(crowns=crown_annotations, m=m, config=config, trainer=trainer)

    if trees is None:
        return None
    
    # Remove predictions for dead trees
    if filter_dead:
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ensembleTaxonID"] = "DEAD"
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_label"] = None
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ens_score"] = None
        
    # Calculate crown area
    trees["crown_area"] = trees.geometry.apply(lambda x: x.area)
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    
    print("{} trees predicted".format(trees.shape[0]))
    
    #Save .shp
    basename = os.path.splitext(os.path.basename(crown_annotations.RGB_tile.unique()[0]))[0]
    trees.to_file(os.path.join(savedir, "{}.shp".format(basename)))
    
    return trees

def predict_crowns(PATH, config):
    """Predict a set of tree crowns from RGB data"""
    m = main.deepforest()
    if torch.cuda.is_available():
        print("CUDA detected")
        m.config["gpus"] = 1
        m.config["workers"] = config["DeepForest_workers"]
        m.config["preload_images"] = config["DeepForest_preload"]
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
    gdf["siteID"] = None
    gdf["box_id"] = None
    gdf["plotID"] = None
    gdf["taxonID"] = None
    gdf["RGB_tile"] = PATH
    
    return gdf

def predict_species(crowns, m, trainer, config):
    """Compute hierarchical prediction without predicting unneeded levels""" 
    config["crop_dir"] = config["prediction_crop_dir"]
    ds = TreeDataset(df=crowns, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))
    if predictions is None:
        return None
    
    results = m.gather_predictions(predictions)
    ensemble_df = m.ensemble(results)
    ensemble_df = results.merge(crowns, on="individual")
            
    return ensemble_df

def predict_dead(crowns, dead_model_path, config):
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    #The batch norm statistics are not helpful in generalization, turn off.
    dead_model.train()
        
    ds = dead.utm_dataset(crowns=crowns, config=config)
    dead_dataloader = dead_model.predict_dataloader(ds)
    trainer = Trainer(gpus=config["gpus"], enable_checkpointing=False)
    outputs = trainer.predict(dead_model, dead_dataloader)
    print("len of predict is {}".format(len(outputs)))
    
    stacked_outputs = np.vstack(np.concatenate(outputs))
    label = np.argmax(stacked_outputs,1)
    score = np.max(stacked_outputs, 1)  
    
    return label, score


