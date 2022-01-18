#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import rasterio
from src import generate
from src.main import TreeModel
from src import data 
from src import start_cluster
from src import neon_paths
import os
from torchvision import transforms
from torch.nn import functional as F
import torch
import numpy as np

def predict_tile(PATH, model_path, config, min_score, taxonIDs, client = None):
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
    trees = predict_species(HSI_path=PATH, crowns=crowns, model_path=model_path, config=config)
    chosen_trees = choose_trees(trees, min_score=min_score, taxonIDs=taxonIDs)
    
    return chosen_trees

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

def create_crops(crowns, config, client = None):
    
    crops = generate.generate_crops(gdf=crowns,
                                    sensor_glob=config["HSI_sensor_pool"],
                                    rgb_glob = config["rgb_sensor_pool"],
                                    convert_h5=config["convert_h5"],
                                    HSI_tif_dir=config["HSI_tif_dir"],
                                    savedir=config["crop_dir"],
                                    client=client)
    crops["individual"] = crops["individualID"]
    crops = crops.merge(crowns[["individual","geometry"]], on="individual")
    
    return crops

def predict_species(crowns, HSI_path, model_path, config):
    m = TreeModel.load_from_checkpoint(model_path)
    src = rasterio.open(HSI_path) 
    preds = []
    scores = []
    for geom in crowns.geometry:
        left, bottom, right, top = geom.bounds
        crop = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)) 
        #preprocess and batch
        image = data.preprocess_image(crop, channel_is_first=True)
        image = transforms.functional.resize(image, size=(config["image_size"],config["image_size"]), interpolation=transforms.InterpolationMode.NEAREST)
        image = torch.unsqueeze(image, dim = 0)
        
        #Classify pixel crops
        m.model.eval() 
        with torch.no_grad():
            class_probs = m.model(image) 
            class_probs = F.softmax(class_probs, 1)
        class_probs = class_probs.detach().numpy()
        index = np.argmax(class_probs)
        label = m.index_to_label[index]
        
        #Average score for selected label
        score = class_probs[:,index]
        
        preds.append(label)
        scores.append(score)
    
    crowns["pred_taxa_top1"] = preds
    crowns["top1_score"] = scores

    return crowns

def choose_trees(trees, min_score, taxonIDs):
    trees = trees[trees.top1_score > min_score]
    trees = trees[trees.pred_taxa_top1.isin(taxonIDs)]
    
    return trees