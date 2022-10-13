#Generate boxes for the DeepForest Zenodo locations
from src import neon_paths
from src import patches
from src import start_cluster
import geopandas as gpd
import pandas as pd
import glob
import os
import random
from distributed import wait

files = glob.glob("/orange/idtrees-collab/zenodo/predictions/*.shp")
random.shuffle(files)

client = start_cluster.start(cpus=50)

def run(f):
    basename = os.path.splitext(os.path.basename(f))[0]
    shp = gpd.read_file(f).sample(frac=1)    
    shp = shp.head(10)
    rgb_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30010.001/**/Camera/**/*.tif", recursive=True)  
    shp["individual"] = ["{}_{}".format(basename, x) for x in shp.index]
    #for index, row in shp.iterrows():
        #sensor_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=row["geometry"].bounds)
        #basename = row["individual"]
        #patches.crop(row["geometry"].bounds, sensor_path, savedir="/orange/idtrees-collab/DeepTreeAttention/crops/zenodo/", basename=basename)
    
    hsi_pool = glob.glob("/orange/ewhite/NeonData/*/DP3.30006.001/**/Reflectance/*.h5", recursive=True)
    for index, row in shp.iterrows():
        sensor_path = neon_paths.lookup_and_convert(hyperspectral_pool=hsi_pool, bounds=row["geometry"].bounds, rgb_pool=rgb_pool, savedir="/orange/idtrees-collab/Hyperspectral_tifs/")
        basename = row["individual"]
        patches.crop(row["geometry"].bounds, sensor_path, savedir="/orange/idtrees-collab/DeepTreeAttention/crops/zenodo/", basename=basename)
    
    as_csv = pd.DataFrame(shp)
    as_csv.to_csv("/orange/idtrees-collab/DeepTreeAttention/crops/zenodo/{}.csv".format(basename))

futures = []
for f in files:
    future = client.submit(run, f)
    futures.append(future)

wait(futures)

