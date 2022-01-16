#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import glob
import rasterio
from src import generate
from src.main import TreeModel
from src.data import TreeDataset
import os
from tempfile import gettempdir
import torch

def predict_tile(PATH, model_path, config, min_score, taxonIDs):
    #get rgb from HSI path
    HSI_basename = os.path.basename(PATH)
    if "hyperspectral" in HSI_basename:
        rgb_name = "{}.tif".format(HSI_basename.split("_hyperspectral")[0])    
    else:
        rgb_name = HSI_basename           
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    rgb_path = [x for x in rgb_pool if rgb_name in x][0]
    crowns = predict_crowns(rgb_path)
    crops = create_crops(crowns, config=config)
    crops["tile"] = PATH
    
    trees = predict_species(crops, model_path=model_path, config=config)
    chosen_trees = choose_trees(trees, min_score=min_score, taxonIDs=taxonIDs)
    
    return chosen_trees

def predict_crowns(PATH):
    m = main.deepforest()
    m.use_release(check_release=False)
    boxes = m.predict_tile(PATH)
    r = rasterio.open(PATH)
    transform = r.transform     
    crs = r.crs
    annotations_to_shapefile(boxes, transform, crs)
    gdf = annotations_to_shapefile(boxes, transform=transform, crs=crs)
    
    #Dummy variables for schema
    gdf["individual"] = range(gdf.shape[0])
    gdf["plotID"] = None
    gdf["siteID"] = None #TODO
    gdf["box_id"] = None
    gdf["plotID"] = None
    gdf["taxonID"] = None
    
    return gdf

def create_crops(crowns, config):
    crops = generate.generate_crops(gdf=crowns,
                                    sensor_glob=config["HSI_sensor_pool"],
                                    rgb_glob = config["rgb_sensor_pool"],
                                    convert_h5=config["convert_h5"],
                                    HSI_tif_dir=config["HSI_tif_dir"],
                                    savedir=config["crop_dir"])
    crops["individual"] = crops["individualID"]
    crops = crops.merge(crowns[["individual","geometry"]], on="individual")
    
    return crops

def predict_species(crops, model_path, config):
    m = TreeModel.load_from_checkpoint(model_path)
    tmpdir = gettempdir()
    basename = os.path.splitext(os.path.basename(crops["tile"].unique()[0]))[0]
    fname = "{}/{}.csv".format(tmpdir, basename)
    crops.to_csv(fname)
    ds = TreeDataset(fname, config=config, train=False)
    data_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["workers"]
    )
    df = m.predict_dataloader(data_loader, train=False)
    df["individual"] = df.individual.astype(int)
    df = df.merge(crops[["individual","geometry"]], on="individual")
    
    return df

def choose_trees(trees, min_score, taxonIDs):
    trees = trees[trees.top1_score > min_score]
    trees = trees[trees.pred_taxa_top1.isin(taxonIDs)]
    
    return trees