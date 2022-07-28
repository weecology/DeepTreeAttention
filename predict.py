from src import predict
from src import data
from src import neon_paths
from glob import glob
import geopandas as gpd
import pandas as pd
from src.start_cluster import start
from distributed import wait, as_completed
import os
import re
import traceback
from pytorch_lightning.loggers import CometLogger

def find_rgb_files(site, config, year="2021"):
    tiles = glob(config["rgb_sensor_pool"], recursive=True)
    tiles = [x for x in tiles if site in x]
    tiles = [x for x in tiles if "neon-aop-products" not in x]
    tiles = [x for x in tiles if "/{}/".format(year) in x]
    
    #Only allow tiles that are within OSBS station boundary
    osbs_tiles = []
    for rgb_path in tiles:
        basename = os.path.basename(rgb_path)
        geo_index = re.search("(\d+_\d+)_image", basename).group(1)
        if ((float(geo_index.split("_")[0]) > 399815.5) &
        (float(geo_index.split("_")[0]) < 409113.7) &
        (float(geo_index.split("_")[1]) > 3282308) &
        (float( geo_index.split("_")[1]) < 3290124)):
            osbs_tiles.append(rgb_path)
            
    return osbs_tiles
    
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

comet_logger = CometLogger(project_name="DeepTreeAttention2", workspace=config["comet_workspace"], auto_output_logging="simple")    
comet_logger.experiment.add_tag("prediction")

gpu_client = start(gpus=7, mem_size="20GB")
cpu_client = start(cpus=5, mem_size="8GB")
species_model_paths = ["/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/24bd8917f519410ab6dd2bd467d42a5c.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/39a4fb41b83047579b5a1de2b0381e28.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/485448fc3c3746389a899f01f08f3bcd.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9b2472ab74644f4eb0e142457ec5b1a1.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/4a83ea2a75994f479448876b1ea19c3b.pt"
                       
                       ]

dead_model_path = "/orange/idtrees-collab/DeepTreeAttention/Dead/snapshots/c4945ae57f4145948531a0059ebd023c.pl"
config["crop_dir"] = "/blue/ewhite/b.weinstein/DeepTreeAttention/67ec871c49cf472c8e1ae70b185addb1"
savedir = config["crop_dir"] 

#Create output folders
for species_model_path in species_model_paths:
    #Save each file seperately in a dir named for the species model
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                  os.path.splitext(os.path.basename(species_model_path))[0])
    try:
        os.mkdir(prediction_dir)
    except:
        pass

#generate HSI_tif data if needed.
h5_pool = glob(config["HSI_sensor_pool"], recursive=True)
h5_pool = [x for x in h5_pool if not "neon-aop-products" in x]
hyperspectral_pool = glob(config["HSI_tif_dir"]+"*")

# Step 1 Find RGB Tiles and convert HSI
tiles = find_rgb_files(site="OSBS", config=config)
tif_futures = cpu_client.map(
    convert,
    tiles,
    hyperspectral_pool=h5_pool,
    savedir=config["HSI_tif_dir"])
wait(tif_futures)

cpu_client.close()
    
predict_futures = []
for species_model_path in species_model_paths:
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                  os.path.splitext(os.path.basename(species_model_path))[0])       
    for x in tiles:
        basename = os.path.splitext(os.path.basename(x))[0]                
        shpname = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/crowns/{}.shp".format(basename)        
        if not os.path.exists(shpname):
            crowns = predict.find_crowns(rgb_path=x, config=config, dead_model_path=dead_model_path)     
            crowns.to_shp(shpname)        
        crowns = gpd.read_file(shpname)
        predict_future = gpu_client.submit(predict.predict_tile,
            crowns=crowns,
            img_pool=hyperspectral_pool,
            filter_dead=True,
            species_model_path=species_model_path,
            savedir=prediction_dir,
            config=config)
        predict_futures.append(predict_future)
    
wait(predict_futures)

for x in predict_futures:
    try:
        predicted_trees = x.result()
    except Exception as e:
        print(e)
        traceback.print_exc()
        continue        
    
#Gather outputs
counts_across_models = []
for species_model_path in species_model_paths:
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/",
                                  os.path.splitext(os.path.basename(species_model_path))[0])    
    input_dir = os.path.join(prediction_dir,"*.shp")
    files = glob(input_dir)
    counts = []
    for x in files:
        gdf = gpd.read_file(x)
        tile_count = gdf.ensembleTa.value_counts()
        counts.append(tile_count)
    
    total_counts = pd.Series()
    for ser in counts:
        total_counts = total_counts.add(ser, fill_value=0)
    
    total_counts.sort_values()
    total_counts.sum()
    total_counts.to_csv("{}/abundance.csv".format(prediction_dir))    
    total_counts["model"] = os.path.basename(species_model_path)
    counts_across_models.append(total_counts)
counts_across_models = pd.concat(counts_across_models)
counts_across_models.to_csv("abundance.csv")
