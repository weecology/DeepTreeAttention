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
gpu_client = start(gpus=2, mem_size="10GB")
cpu_client = start(cpus=10, mem_size="8GB")
species_model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/1afdb5de011e4c1d8419a904e42d40bc.pl"
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/91ba2dc9445547f48805ec60be0a2f2f"
savedir = config["crop_dir"] 

#generate HSI_tif data if needed.
hyperspectral_pool = glob(config["HSI_sensor_pool"], recursive=True)
hyperspectral_pool = [x for x in hyperspectral_pool if not "neon-aop-products" in x]
annotation_path = None

# Step 1 Find RGB Tiles and convert HSI
if annotation_path is None:
    tiles = find_rgb_files(site="OSBS", config=config)[:2]
    tif_futures = cpu_client.map(convert, tiles, hyperspectral_pool=hyperspectral_pool, savedir=config["HSI_tif_dir"])
    wait(tif_futures)
    
    # Step 2 - Predict Crowns
    crown_futures = gpu_client.map(
        predict.find_crowns,        
        tiles,         
        config=config,
        dead_model_path=dead_model_path
    )
    
    # Step 3 - Crop Crowns
    crop_futures = []
    for x in as_completed(crown_futures):
        crowns = x.result()
        crop_future = cpu_client.submit(
            predict.generate_crops,
            crowns=crowns[:10],
            config=config,
            dead_model_path=dead_model_path)
        crop_futures.append(crop_future)
        basename = os.path.splitext(os.path.basename(crowns.RGB_tile.unique()[0]))[0]
        crowns.to_file("{}/crowns_{}.shp".format(savedir,basename))
    
    wait(crop_futures)
    
    for x in crop_futures:
        annotations = x.result()
        annotations.to_csv("{}/annotations_{}.csv".format(savedir, annotations.RGB_tile.unique()[0]))
else:
    pass
 
df = dataframe.read_csv(savedir,"annotations*.csv")

#Save each file seperately in a dir named for the species model
savedir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",os.path.basename(species_model_path))
try:
    os.mkdir(savedir)
except:
    pass

geo_index = [re.search("(\d+_\d+)_image", os.path.basename(x)).group(1) for x in hsi_tifs]
df["geo_index"] = df.RGB_tile.apply(lambda x: re.search("(\d+_\d+)_image", os.path.basename(x)).group(1))
geo_indices = df.geo_index.unique()

futures = []
for geo_index in geo_indices:
    tile_crowns = df[df.geo_index == geo_index]
    crowns = gpd.read_file("{}/crowns_{}.shp".format(savedir, tile_crowns.RGB_tile.unique[0]))
    
    future = gpu_client.submit(
        predict.predict_tile,
        annotations=tile_crowns,
        crowns=crowns,
        filter_dead=True,
        species_model_path=species_model_path,
        config=config,
        savedir=savedir
    )
    futures.append(future)

wait(futures)

for future in futures:
    try:
        trees = future.result()    
    except Exception as e:
        print(e)
        print(traceback.print_exc())

