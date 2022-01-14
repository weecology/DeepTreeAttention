#Predict
from deepforest import main
from deepforest.utilities import annotations_to_shapefile
import rasterio
from src import generate
from src.main import TreeModel
from src.data import TreeDataset
import os
from tempfile import gettempdir
import torch

def predict_tile(PATH, model_path, config):
    crowns = predict_crowns(PATH)
    crops = create_crops(crowns, config=config)
    trees = predict_species(crops, model_path)
    chosen_trees = choose_trees(trees)
    
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
    gdf["individual"] = None
    gdf["plotID"] = None
    gdf["siteID"] = None #TODO
    gdf["box_id"] = None
    gdf["plotID"] = None

    return gdf

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
    df = m.predict_dataloader(data_loader)
    
    return df
    

def create_crops(crowns, config):
    crops = generate.generate_crops(gdf=crowns,
                                    sensor_glob=config["HSI_sensor_pool"],
                                    rgb_glob = config["rgb_sensor_pool"],
                                    convert_h5=config["convert_h5"],
                                    HSI_tif_dir=["HSI_tif_dir"],
                                    savedir=config["crop_dir"])
    
    return crops

def choose_trees(trees):
    trees = trees[trees.score > 0.7]
    trees = trees[trees.pred_taxa_top1.isin(["PICL","MAGNO","CAGL8"])]
    
    return trees