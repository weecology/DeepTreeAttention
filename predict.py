
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

from src import predict
from src import data
from src import neon_paths
from src.utils import create_glob_lists 
from src.models import multi_stage

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
        year = path.split("/")[6]
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
comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")    

comet_logger.experiment.log_parameters(config)

#client = Client()
client = start(cpus=1, mem_size="10GB")

#Get site arg
site=str(sys.argv[1])
comet_logger.experiment.add_tag("prediction_{}".format(site))
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops"
savedir = config["crop_dir"] 

species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/000b1ecf0ca6484893e177e3b5d42c7e_NIWO.pt",
    "RMNP": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9d71632542494af494c83fb4487747ce_RMNP.pt",    
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ecfdd5bf772a40cab89e89fa1549f13b_SJER.pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/686204cb0d5343b0b20613a6cf25f69b_WREF.pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/20ce0ca489444e84997e82b4b293e86c_SERC.pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/127191a057e64a7fb7267cc889e56c25_GRSM.pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0e3178ac37434aeb90ac207c18a9caf7_DEJU.pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/152a61614a4a48cf84b27f5880692230_BONA.pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/990e6f1101b2423f86d4cd16f373deab_TREE.pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/e5581d389ff34dbeb406261a6e512141_STEI.pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/185caf9f910c4fd3a7f5e470b6828090_UNDE.pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/97b895af32014434ac5acb8b1eb59292_DELA.pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/537b80fd46e64c77a1c367dcbef713e3_LENO.pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/00fef05fa70243f1834ee437406150f7_OSBS.pt",
    "JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/86d51ae4b7a34308bc99c19f8eeadf41_JERC.pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c8c741de6f1e442aaf3c8c2c3e323b9b_TALL.pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/dfa2c966002c498a8a8f760b179a388c_CLBJ.pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ca17bf7c36fe42e6bd83a358243c012b_TEAK.pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/97695ef8ec6a481fb3515d29d2cf33bb_SOAP.pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/f2c069f59b164795af482333a5e7fffb_YELL.pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b5efc0037529431092db587727fb4fe9_MLBS.pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/34d4b60b7f9e40cb83c15697bc754f30_BLAN.pt",
    "UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/084b83c44d714f23b9d96e0a212f11f1_UKFS.pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/01bfce34aacd455bb1a4b4d80deb16d2_BART.pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/973a01b0c1a349ebaf4fc8454ffc624d_HARV.pt"}

def create_landscape_map(site, model_path, config, client, rgb_pool, hsi_pool, h5_pool, CHM_pool):
    #Prepare directories
    # Crop Predicted Crowns
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    try:
        prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}".format(site, model_name))
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
    for year in [2022, 2021, 2020, 2019]:
        tiles = find_rgb_files(site=site, rgb_pool=rgb_pool, year=year)
        if len(tiles) > 0:
            break
    
    # remove existing files
    #tarfiles = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/tar/*.tar*".format(site))
    #tiles_to_run = []
    #for tile in tiles:
        #image_name = os.path.splitext(os.path.basename(tile))[0]
        #needs_to_be_run = np.sum([image_name in x for x in tarfiles]) == 0
        #if needs_to_be_run:
            #tiles_to_run.append(tile)
        
    if len(tiles) == 0:
        raise ValueError("There are no RGB tiles left to run for any year since 2019 for {}".format(site))
    
    #tif_futures = client.map(
        #convert,
        #tiles,
        #hyperspectral_pool=h5_pool,
        #savedir=config["HSI_tif_dir"]
    #)
    #wait(tif_futures)
    
    species_futures = []
    crop_futures = []
    
    # Predict crowns
    for x in tiles:
        basename = os.path.splitext(os.path.basename(x))[0]
        crop_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/{}/{}".format(site, basename)
        try:
            os.mkdir(crop_dir)
        except:
            pass
        try:
            crown_path = predict.find_crowns(
                rgb_path=x,
                config=config,
                dead_model_path=dead_model_path,
                savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns",
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
    m = multi_stage.MultiStage.load_from_checkpoint(model_path, config=config)
    trainer = Trainer()
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
                config=config)
        except:
            traceback.print_exc()
            continue
            
    return crop_futures
            
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)
futures = create_landscape_map(
    site,
    species_model_paths[site],
    config,
    client, 
    rgb_pool=rgb_pool,
    h5_pool=h5_pool,
    hsi_pool=hsi_pool,
    CHM_pool=CHM_pool)
