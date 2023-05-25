#Predict
import os
import geopandas as gpd
import rasterio
import numpy as np
from torchvision import transforms
import torch
import tarfile
import shutil
import pandas as pd
from pytorch_lightning import Trainer

from deepforest import main
from deepforest.utilities import annotations_to_shapefile

from src.models import dead
from src.CHM import postprocess_CHM
from src.generate import generate_crops
from src.models.multi_stage import TreeDataset, MultiStage
from src.data import __file__

ROOT = os.path.dirname(os.path.dirname(__file__))

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
    
def find_crowns(rgb_path, config, dead_model_path=None, savedir=None, CHM_pool=None, overwrite=False):
    """Predict deepforest crowns"""
    basename = os.path.splitext(os.path.basename(rgb_path))[0]  
    output_filename = "{}/{}.shp".format(savedir, basename)
    
    if not overwrite:
        if os.path.exists(output_filename):
            return output_filename
        
    crowns = predict_crowns(rgb_path, config)
    if crowns is None:
        return None
    crowns["tile"] = rgb_path
    
    #CHM filter
    if CHM_pool:
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
    
    if savedir:
        basename = os.path.splitext(os.path.basename(rgb_path))[0]  
        output_filename = "{}/{}.shp".format(savedir, basename)
        filtered_crowns.to_file(output_filename)
        return output_filename
    else:
        return filtered_crowns

def generate_prediction_crops(crown_path, config, rgb_pool, h5_pool, img_pool, crop_dir, client=None, as_numpy=True, overwrite=False):
    """Create prediction crops for model.predict"""
    basename = os.path.splitext(os.path.basename(crown_path))[0]
    dirname = os.path.dirname(crop_dir)
    output_name = "{}/shp/{}.shp".format(dirname, basename)
    
    if overwrite is False:
        if os.path.exists(output_name):
            return output_name
        else:
            crowns = gpd.read_file(crown_path)            
    else:
        crowns = gpd.read_file(crown_path)
        
    crown_annotations = generate_crops(
        crowns,
        savedir=crop_dir,
        img_pool=img_pool,
        h5_pool=h5_pool,
        convert_h5=config["convert_h5"],   
        rgb_pool=rgb_pool,
        HSI_tif_dir=os.environ["TMPDIR"],
        client=client,
        as_numpy=as_numpy
    )
    
    if crown_annotations.empty:
        print("No annotations created")
        return None
    
    #Write file alongside       
    crown_annotations = gpd.GeoDataFrame(crown_annotations, geometry="geometry")    
    crown_annotations = crown_annotations.merge(crowns[["individual","dead_label","dead_score", "score"]])
    
    crown_annotations.to_file(output_name)  
    
    # Tar each archive.
    tar_name = "{}/tar/{}.tar.gz".format(dirname, basename)
    with tarfile.open(tar_name,"w") as tfile:
        for path in crown_annotations.image_path:
            filename = "{}/{}".format(crop_dir, path)
            tfile.add(filename, arcname=path)
            os.remove(filename)
    
    return output_name

def predict_tile(crown_annotations, m, config, savedir, site, trainer, filter_dead=False):
    """Predict a set of crown labels from a annotations.shp
    Args:
        crown_annotations: path .shp from predict.generate_prediction_crops
        m: pytorch model to predict data
        config: config.yml
        savedir: directory to save tile predictions
        filter_dead: filter dead model
    """
    
    # When specifying a tarfile, we save crops into local storage
    config["crop_dir"] = os.path.join(os.environ["TMPDIR"],site)
    os.makedirs(config["crop_dir"], exist_ok=True)
    config["pretrained_state_dict"] = None 
    
    tarfilename = crown_annotations.replace("shp","tar")
    tarfilename = "{}.tar.gz".format(os.path.splitext(tarfilename)[0])
    with tarfile.open(tarfilename, 'r') as archive:
        archive.extractall(config["crop_dir"])   
        
    trees = predict_species(crowns=crown_annotations, m=m, config=config, trainer=trainer)

    if trees is None:
        return None
    
    # Remove predictions for dead trees
    if filter_dead:
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"pred_taxa_top1"] = "DEAD"
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"label"] = None
        trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"score"] = None
        
    # Calculate crown area
    trees["crown_area"] = trees.geometry.apply(lambda x: x.area)
    print("{} trees predicted".format(trees.shape[0]))
    
    #site ID
    trees["siteID"] = site
    
    #Merge taxonID and clean up columns
    neon_taxonID = pd.read_csv("{}/data/raw/OS_TAXON_PLANT-20220330T142149.csv".format(ROOT))
    neon_taxonID = neon_taxonID[["taxonID","scientificName"]]
    trees["taxonID"] = trees["ensembleTaxonID"]
    trees = trees.merge(neon_taxonID, on="taxonID")
    trees["crown_score"] = trees["score"]
    trees = trees.drop(columns=["pred_taxa_top1","label","score","taxonID"])
    trees = trees.groupby("individual").apply(lambda x: x.head(1)).reset_index(drop=True)
    trees = gpd.GeoDataFrame(trees, geometry="geometry")    
    
    #Save .shp
    output_name = os.path.splitext(os.path.basename(crown_annotations))[0]
    trees.to_file(os.path.join(savedir, "{}.shp".format(output_name)))
    
    #remove files
    shutil.rmtree(config["crop_dir"])
    
    return trees

def predict_crowns(PATH, config):
    """Predict a set of tree crowns from RGB data"""
    m = main.deepforest()
    if torch.cuda.is_available():
        print("CUDA detected")
        m.config["gpus"] = 1
        m.config["workers"] = config["DeepForest_workers"]
        m.config["preload_images"] = config["DeepForest_preload"]
        m.config["batch_size"] = config["DeepForest_batch_size"]
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
    
    # Make sure dask worker doesn't hold on to memory.
    del m
    torch.cuda.empty_cache()
    
    return gdf

def predict_species(crowns, m, trainer, config):
    """Compute hierarchical prediction without predicting unneeded levels""" 
    crowns = gpd.read_file(crowns)
    ds = TreeDataset(df=crowns, train=False, config=config)
    predictions = trainer.predict(m, dataloaders=m.predict_dataloader(ds))    
    results = m.gather_predictions(predictions)
    results = results.merge(crowns[["geometry","individual","taxonID","siteID","dead_label","dead_score", "score"]], on="individual", how="right")
    ensemble_df = m.ensemble(results)
    ensemble_df = gpd.GeoDataFrame(ensemble_df, geometry="geometry")
    
    return ensemble_df

def predict_dead(crowns, dead_model_path, config):
    """Classify RGB as alive or dead"""
    dead_model = dead.AliveDead.load_from_checkpoint(dead_model_path, config=config)
    
    # The batch norm statistics are not helpful in generalization, turn off.
    dead_model.train()
        
    ds = dead.utm_dataset(crowns=crowns, config=config)
    dead_dataloader = dead_model.predict_dataloader(ds)
    trainer = Trainer(enable_checkpointing=False)
    outputs = trainer.predict(dead_model, dead_dataloader)
    print("len of predict is {}".format(len(outputs)))
    
    stacked_outputs = np.vstack(np.concatenate(outputs))
    label = np.argmax(stacked_outputs,1)
    score = np.max(stacked_outputs, 1)  
    
    #Make dask doesn't hold on to memory
    del dead_model
    torch.cuda.empty_cache()
    
    return label, score


