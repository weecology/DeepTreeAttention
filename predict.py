#OSBS mining
from src import predict
from src import data
from src import neon_paths
from glob import glob
import pandas as pd
import geopandas as gpd
from src.start_cluster import start
from distributed import wait

def find_files(site, year, config):
    tiles = glob(config["HSI_sensor_pool"] + "*.tif")
    tiles = [x for x in tiles if site in x]
    tiles = [x for x in tiles if "/{}/".format(year) in x]
    
    return tiles
    
config = data.read_config("config.yml")
model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9545b1fc496b45eeb6267f7ea7575f4d.pl"

predictions = []
tiles = find_files(site="OSBS", config=config)

#generate HSI_tif data if needed.
hyperspectral_pool = glob.glob(config["HSI_sensor_pool"])
rgb_pool = glob.glob(config["rgb_sensor_pool"])

cpu_client = start(cpus=50)
gpu_client = start(gpus=5, mem_size="50GB")

tif_futures = cpu_client.map(neon_paths.lookup_and_convert, tiles, hyperspectral_pool=hyperspectral_pool, rgb_pool=rgb_pool, savedir = config["HSI_tif_dir"])
wait(tif_futures)
cpu_client.close()

futures =  []
for x in tiles:
    future = gpu_client.submit(predict.predict_tile, x, model_path=model_path, config=config, min_score=0.7, taxonIDs=["PICL","MAGNO","CAGL8"])
    futures.append(future)

wait(futures)

for future in futures:
    try:
        trees = future.result()
        if not trees.empty:
            predictions.append(trees)        
    except:
        pass

predictions = pd.concat(predictions)
predictions = gpd.GeoDataFrame(predictions, geometry="geometry")
predictions.to_file("results/OSBS_predictions.shp")
