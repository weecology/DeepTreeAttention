
import geopandas as gpd
import traceback
from src.start_cluster import start
from distributed import wait, as_completed, fire_and_forget, Client
import os
import numpy as np
import argparse
import re
from pytorch_lightning.loggers import CometLogger
from pytorch_lightning import Trainer
import sys
import random
import copy

from src import predict
from src import data
from src import neon_paths
from src.utils import create_glob_lists 
from src.models import multi_stage
from src.model_list import species_model_paths

def find_rgb_files(site, rgb_pool, year="2020"):
    # TREE and STEI are the same site.
    if site == "TREE":
        site = "STEI"
        
    tiles = [x for x in rgb_pool if site in x]
    tiles = [x for x in tiles if "/{}/".format(year) in x]
    
    return tiles

def convert(rgb_path, hyperspectral_pool, savedir):
    #convert .h5 hyperspec tile if needed
    basename = os.path.basename(rgb_path)
    geo_index = re.search("(\d+_\d+)_image", basename).group(1)
    h5_list = [x for x in hyperspectral_pool if geo_index in x]
    tif_paths = []
    for path in h5_list:
        year = path.split("/")[7]
        tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral_{}.tif".format(year)
        tif_path = "{}/{}".format(savedir, tif_basename)
        if not os.path.exists(tif_path):
            tif_paths.append(neon_paths.convert_h5(path, rgb_path, savedir, year=year))
        else:
            tif_paths.append(tif_path)
    
    return tif_paths

#Params
config = data.read_config("config.yml")
config["preload_images"] = False
config["preload_image_dict"] = False

comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")    

comet_logger.experiment.log_parameters(config)

#client = Client()
client = start(cpus=10, mem_size="5GB")

#Get site arg
site=str(sys.argv[1])
comet_logger.experiment.add_tag("prediction_{}".format(site))
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops"
savedir = config["crop_dir"] 

def create_landscape_map(site, model_path, config, client, rgb_pool, hsi_pool, h5_pool, CHM_pool):
    #Prepare directories
    # Crop Predicted Crowns
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    try:
        prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/2021/predictions/{}/{}".format(site, model_name))
        os.makedirs(prediction_dir, exist_ok=True)        
    except:
        pass
    
    try:
        os.mkdir("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}".format(site))
        os.mkdir("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/tar".format(site))
        os.mkdir("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/shp".format(site))
    except:
        pass

    ### Step 1 Find RGB Tiles and convert HSI, prioritize 2022
    for year in [2019]:
        tiles = find_rgb_files(site=site, rgb_pool=rgb_pool, year=year)
        if len(tiles) > 0:
            break
    
    tiles = tiles[:10]
    if len(tiles) == 0:
        raise ValueError("There are no RGB tiles left to run for any year since 2019 for {}".format(site))
    
    #tif_futures = client.map(
    #    convert,
    #    tiles,
    #    hyperspectral_pool=h5_pool,
    #   savedir=config["HSI_tif_dir"])
    #wait(tif_futures)
    
    species_futures = []
    crop_futures = []
    
    # Randomize tiles
    random.shuffle(tiles)
    # Predict crowns
    for x in tiles:
        basename = os.path.splitext(os.path.basename(x))[0]
        crop_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/year/2021/site_crops/{}/{}".format(site, basename)
        try:
            os.mkdir(crop_dir)
        except:
            pass
        try:
            crown_path = predict.find_crowns(
                rgb_path=x,
                config=config,
                dead_model_path=dead_model_path,
                savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/year/2021/crowns",
                overwrite=False)
        except:
            traceback.print_exc()

        crop_future = client.submit(
            predict.generate_prediction_crops,
            crown_path,
            config,
            crop_dir=crop_dir,
            as_numpy=True,
            client=None,
            img_pool=hsi_pool,
            h5_pool=h5_pool,
            rgb_pool=rgb_pool,
            overwrite=False
        )
        crop_futures.append(crop_future)
    
    # load model
    # Hot fix for several small sites that were better in hierarchical models
    site_config = copy.deepcopy(config)
    if any(x in model_path for x in ["SJER","WREF","YELL"]):
        site_config["max_flat_species"] = 0

    m = multi_stage.MultiStage.load_from_checkpoint(model_path, config=site_config)
    trainer = Trainer(devices=config["gpus"])
    for finished_crop in as_completed(crop_futures):
        try:
            crown_annotations_path = finished_crop.result()
        except:
            traceback.print_exc()
            continue
        if crown_annotations_path is None:
            continue
        print(crown_annotations_path)

        output_name = os.path.splitext(os.path.basename(crown_annotations_path))[0]
        output_name = os.path.join(prediction_dir, "{}.shp".format(output_name))
        
        if os.path.exists(output_name):
            print("{} exists".format(output_name))
            continue
        try:
            species_prediction = predict.predict_tile(
                crown_annotations=crown_annotations_path,
                filter_dead=True,
                trainer=trainer,
                m=m,
                savedir=prediction_dir,
                site=site,
                config=site_config)
        except:
            traceback.print_exc()
            continue
            
    return crop_futures
            
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config, year="2021")

futures = create_landscape_map(
    site,
    species_model_paths[site],
    config,
    client=client, 
    rgb_pool=rgb_pool,
    h5_pool=h5_pool,
    hsi_pool=hsi_pool,
    CHM_pool=CHM_pool)
