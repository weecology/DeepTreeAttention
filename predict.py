from src import predict
from src import data
from src import neon_paths
from glob import glob
import geopandas as gpd
import pandas as pd
from src.start_cluster import start
from distributed import wait, as_completed
from dask import dataframe
import os
import re
import traceback

def find_rgb_files(site, config, year="2021"):
    tiles = glob(config["rgb_sensor_pool"], recursive=True)
    tiles = [x for x in tiles if site in x]
    tiles = [x for x in tiles if "neon-aop-products" not in x]
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
#No daemonic dask children
config = data.read_config("config.yml")
config["workers"] = 0
config["preload_images"] = False 

#gpu_client = start(gpus=2, mem_size="10GB")
cpu_client = start(cpus=10, mem_size="8GB")
species_model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt"
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/67ec871c49cf472c8e1ae70b185addb1"
savedir = config["crop_dir"] 

#Save each file seperately in a dir named for the species model
prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                              os.path.splitext(os.path.basename(species_model_path))[0])
try:
    os.mkdir(prediction_dir)
except:
    pass
    
#generate HSI_tif data if needed.
hyperspectral_pool = glob(config["HSI_sensor_pool"], recursive=True)
hyperspectral_pool = [x for x in hyperspectral_pool if not "neon-aop-products" in x]

# Step 1 Find RGB Tiles and convert HSI
tiles = find_rgb_files(site="OSBS", config=config)[:1]
tif_futures = cpu_client.map(convert, tiles, hyperspectral_pool=hyperspectral_pool, savedir=config["HSI_tif_dir"])
wait(tif_futures)

# Step 2 - Predict Crowns
crown_futures = gpu_client.map(
    predict.find_crowns,        
    tiles,         
    config=config,
    dead_model_path=dead_model_path,
)
    
# Step 3 - Crop Crowns
crop_futures = []
for x in as_completed(crown_futures):
    try:
        crowns = x.result()
        ensemble_df = predict.predict_tile(
            crowns=crowns,
            img_pool=hyperspectral_pool,
            filter_dead=True,
            species_model_path=species_model_path,
            savedir=prediction_dir,
            config=config)
    except Exception as e:
        print(e)
        continue
    if crowns is None:
        continue

