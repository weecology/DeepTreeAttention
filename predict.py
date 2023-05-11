
import geopandas as gpd
import traceback
from src.start_cluster import start
from distributed import wait, as_completed, fire_and_forget
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

client = start(cpus=3, mem_size="10GB")

#Get site arg
site=str(sys.argv[1])
comet_logger.experiment.add_tag("prediction_{}".format(site))
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops"
savedir = config["crop_dir"] 

species_model_paths = {
    "NIWO": "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/287f10349eca4497957a03cf0d48b468_NIWO.pt",
    "SJER":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/702f6a7cf1b24307b8a23e25148f7559_SJER.pt",
    "WREF":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/5d96ad72d18549d5ad200e3ad44aa429_WREF.pt",
    "SERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/920a0d718f894963a961437622be3a97_['SERC', 'GRSM'].pt",
    "GRSM":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/422f3b1af269499dac4478619ec5f488_GRSM.pt",
    "DEJU":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0e3178ac37434aeb90ac207c18a9caf7_DEJU.pt",
    "BONA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a86cdf52b3d14568b2d7574a13185868_['BONA', 'DEJU'].pt",
    "TREE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/a1bfa10f43f54f1e82f09b46488dcdac_TREE.pt",
    "STEI":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/7d69a3ee5c6241e8b48cdf09391e0bff_STEI.pt",
    "UNDE":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6676d7c457404a568838eb76dbe9b6f5_UNDE.pt",
    "DELA":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/299e7aa3bdae413a9931542310da9d96_DELA.pt",
    "LENO":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6b81242c32b34ee78273635b372f3186_LENO.pt",
    "OSBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9ee56b8d1b72491c913935d40d9be810_OSBS.pt",
    #"JERC":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/7c80b43d39e54bff955f4de68f548370_['OSBS', 'JERC', 'TALL', 'DSNY'].pt",
    "TALL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/3087af0dcc3c486db17a4a922c43eae9_TALL.pt",
    "CLBJ":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/83c1d8fd4c69479185ed3224abb6e8f9_CLBJ.pt",
    "TEAK":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b9b6882705d24fe6abf12282936ababb_TEAK.pt",
    "SOAP":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/8c3dd91e61e948c0809a46d6a824afc2_SOAP.pt",
    "YELL":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/27765ac5b3754d32bc2c626e257c296f_YELL.pt",                       
    "MLBS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/e1770318881b4eb88a4c975e287573c5_MLBS.pt",
    "BLAN":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/166a8445e3304fc7840aef9ad113bfff_BLAN.pt",
    #"UKFS":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9af9ba5a9e1148daa365d3c893cde875_['MLBS','BLAN','SCBI','UKFS'].pt",
    "BART":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b6296f0f28ea45459df495185b5f9c33_BART.pt",
    "HARV":"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/69b49e1ea47e40d79760e564cbb7bd6e_HARV.pt"}

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



    
