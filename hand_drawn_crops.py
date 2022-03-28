#Crop test data
import glob
import os
import pandas as pd
from src import patches
from src.data import read_config
from src import neon_paths
import geopandas as gpd

config = read_config("config.yml")
test = pd.read_csv(os.path.join(config["data_dir"], config["use_data_commit"], "test.csv"))
crowns = gpd.read_file(os.path.join(config["data_dir"], config["use_data_commit"], "crowns.shp"))
crowns = crowns[crowns.individual.isin(test.individualID)]
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
for index, row in crowns.iterrows():
    sensor_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=row["geometry"].bounds)
    basename = row["individual"]
    bounds = row["geometry"].buffer(5).bounds
    patches.crop(bounds, sensor_path, savedir="/orange/idtrees-collab/DeepTreeAttention/hand_annotations/", basename=basename)

