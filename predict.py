from src import predict
from src import data
from src import neon_paths
from glob import glob
import pandas as pd
import geopandas as gpd
from src.start_cluster import start
from distributed import wait
import os
import re
import traceback


def find_rgb_files(site, config, year=None):
    tiles = glob(config["rgb_sensor_pool"], recursive=True)
    tiles = [x for x in tiles if site in x]
    
    if year:
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
    
    return tif_paths

config = data.read_config("config.yml")
tiles = find_rgb_files(site="OSBS", config=config, year=2021)

#generate HSI_tif data if needed.
hyperspectral_pool = glob(config["HSI_sensor_pool"], recursive=True)
rgb_pool = glob(config["rgb_sensor_pool"], recursive=True)

cpu_client = start(cpus=50, mem_size="8GB")

tif_futures = cpu_client.map(convert, tiles[:1], hyperspectral_pool=hyperspectral_pool, savedir = config["HSI_tif_dir"])
wait(tif_futures)

species_model_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/91ba2dc9445547f48805ec60be0a2f2f"
dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
hsi_tifs = []
for x in tif_futures:
    try:
        for path in x.result():
            hsi_tifs.append(path)
    except Exception as e:
        print(e)
        pass

cpu_client.close()    
gpu_client = start(gpus=12, mem_size="10GB")

#No daemonic dask children
config["workers"] = 0
futures =  []

#Save each file seperately in a dir named for the species model
savedir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",os.path.basename(species_model_dir))
try:
    os.mkdir(savedir)
except:
    pass

geo_index = [re.search("(\d+_\d+)_image", os.path.basename(x)).group(1) for x in hsi_tifs]

for i in pd.Series(geo_index).unique():
    HSI_paths = {}
    tiles = [x for x in hsi_tifs if i in x] 
    for tile in tiles:
        year = os.path.splitext(tile)[0].split("_")[-1]
        HSI_paths[year] = tile
    future = gpu_client.submit(
        predict.predict_tile,
        HSI_paths,
        dead_model_path=dead_model_path,
        species_model_dir=species_model_dir,
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
