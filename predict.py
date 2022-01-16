#OSBS mining
from src import predict
from src import data
from glob import glob
import pandas as pd
import geopandas as gpd
from src.start_cluster import start
from distributed import wait

def find_files(site, config):
    tiles = glob(config["HSI_tif_dir"] + "*.tif")
    tiles = [x for x in tiles if site in x]
    
    return tiles
    
config = data.read_config("config.yml")
model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/9545b1fc496b45eeb6267f7ea7575f4d.pl"

predictions = []
tiles = find_files(site="OSBS", config=config)
cluster = start(gpus=5, mem_size="50GB")
futures = []
for x in tiles:
    future = cluster.submit(predict.predict_tile, x, model_path=model_path, config=config, min_score=0.7, taxonIDs=["PICL","MAGNO","CAGL8"])
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
