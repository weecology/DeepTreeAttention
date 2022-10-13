#Crop test data
import glob
import os
import pandas as pd
from src import patches
from src.data import read_config
from src import neon_paths
import geopandas as gpd

config = read_config("config.yml")
test = pd.read_csv("/orange/idtrees-collab/DeepTreeAttention/crops/c75914da262947709f47c0f1e328f845/test.csv")
crowns = gpd.read_file(os.path.join("/orange/idtrees-collab/DeepTreeAttention/crops/c75914da262947709f47c0f1e328f845/", "crowns.shp"))
crowns = crowns[crowns.individual.isin(test.individualID)]
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
for index, row in crowns.iterrows():
    sensor_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=row["geometry"].bounds)
    basename = row["individual"]
    bounds = row["geometry"].buffer(3).bounds
    patches.crop(bounds, sensor_path, savedir="/orange/idtrees-collab/DeepTreeAttention/crops/test_rgb/", basename=basename)


hsi_pool = glob.glob(config["HSI_sensor_pool"], recursive=True)
for index, row in crowns.iterrows():
    sensor_path = neon_paths.lookup_and_convert(hyperspectral_pool=hsi_pool, bounds=row["geometry"].bounds, rgb_pool=rgb_pool, savedir=config["HSI_tif_dir"])
    basename = row["individual"]
    bounds = row["geometry"].buffer(3).bounds
    patches.crop(bounds, sensor_path, savedir="/orange/idtrees-collab/DeepTreeAttention/crops/test_HSI/", basename=basename)
    