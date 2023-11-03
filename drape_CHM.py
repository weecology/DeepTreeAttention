# Drape predictions
from src import CHM
from src.data import read_config
from src.utils import create_glob_lists
from src.neon_paths import find_sensor_path
from src.start_cluster import start
from src.model_list import species_model_paths

import glob
import geopandas as gpd
import rasterstats
import os
from distributed import wait, as_completed
import traceback

config = read_config("config.yml")
rgb_pool, h5_pool, hsi_pool, CHM_pool = create_glob_lists(config)

all_files = []
for site in species_model_paths:
    model_path = species_model_paths[site]
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, model_name))
    files = glob.glob(prediction_dir, recursive=True)
    all_files.append(files)

all_files = [item for sublist in all_files for item in sublist]
print("Found {} files".format(len(all_files)))

client = start(cpus=10,mem_size="20GB")

def drape(shp, config, CHM_pool):
    """Take in a predictions shapefile and extract CHM height"""

    dirname = os.path.dirname(shp)
    basename = os.path.basename(shp)
    dst = os.path.join(dirname,"draped")
    
    if os.path.exists(dst):
        return dst

    df = gpd.read_file(shp)

    #buffer slightly, CHM model can be patchy
    geom = df.geometry.buffer(1)
    try:
        CHM_path = find_sensor_path(lookup_pool=CHM_pool, bounds=df.total_bounds)
    except ValueError:
        return None
    
    draped_boxes = rasterstats.zonal_stats(geom,
                                            CHM_path,
                                           add_stats={'q99': CHM.non_zero_99_quantile})
    df["height"] = [x["q99"] for x in draped_boxes]
    filtered_trees = df[df["height"]>3]


    os.makedirs(dst, exist_ok=True)
    filtered_trees.to_file(os.path.join(dst, basename))

    return dst

futures = []
for f in all_files:
    future = client.submit(drape, f, CHM_pool=CHM_pool, config=config)
    futures.append(future)

for x in as_completed(futures):
    try:
        print(x.result())
    except:
        traceback.print_exc()

#for f in files:
#    dst = drape(f, config=config, CHM_pool=CHM_pool)
#    print(dst)