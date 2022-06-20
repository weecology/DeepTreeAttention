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

gpu_client = start(gpus=2, mem_size="10GB")
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
regenerate = True
overwrite = False

# Step 1 Find RGB Tiles and convert HSI
if regenerate:
    tiles = find_rgb_files(site="OSBS", config=config)
    if not overwrite:
        tiles = [x for x in tiles if not os.path.exists("{}/crowns_{}.shp".format(savedir, os.path.splitext(os.path.basename(x))[0]))]
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
        crowns = x.result()
        if crowns is None:
            continue
        crop_future = cpu_client.submit(
            predict.generate_crops,
            crowns=crowns,
            config=config,
            dead_model_path=dead_model_path)
        crop_futures.append(crop_future)
        basename = os.path.splitext(os.path.basename(crowns.RGB_tile.unique()[0]))[0]
        crowns.to_file("{}/crowns_{}.shp".format(savedir,basename))
    
    wait(crop_futures)
    
    for x in crop_futures:
        annotations = x.result()
        basename = os.path.splitext(os.path.basename(annotations.RGB_tile.unique()[0]))[0]
        annotations.to_csv("{}/annotations_{}.csv".format(savedir, basename))
else:
    pass

annotation_files = glob(savedir + "/annotations*.csv")

def species_wrapper(annotations_path, species_model_path, config, data_dir, prediction_dir):
    annotations = pd.read_csv(annotations_path)
    basename = os.path.splitext(os.path.basename(annotations.RGB_tile.unique()[0]))[0]    
    crowns = gpd.read_file("{}/crowns_{}.shp".format(data_dir, basename))
    ensemble_df = predict.predict_tile(crowns, annotations, species_model_path, config, prediction_dir, filter_dead=True)
    basename = os.path.splitext(os.path.basename(annotations.RGB_tile.unique()[0]))[0]    
    ensemble_df[["geometry","individual","ens_score",
                 "ensembleTaxonID","crown_area","CHM_height","dead_label","dead_score","RGB_tile"]].to_file("{}/{}.shp".format(prediction_dir, basename))
    
    return ensemble_df

# Without dask
for x in annotation_files[:2]:
    species_wrapper(
        annotations_path=x, 
        species_model_path=species_model_path,
        config=config,
        data_dir=savedir,
        prediction_dir=prediction_dir)

#futures = gpu_client.map(
    #species_wrapper, 
    #annotation_files, 
    #species_model_path=species_model_path, 
    #config=config, 
    #data_dir=savedir,
    #prediction_dir=prediction_dir)

#for future in as_completed(futures):
    #try:
        #trees = future.result()    
    #except Exception as e:
        #print(e)
        #print(traceback.print_exc())

