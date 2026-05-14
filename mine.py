import glob
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio as rio
from src import patches
from src import neon_paths
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data import read_config

config = read_config("config.yml")
shapefiles = glob.glob("/orange/idtrees-collab/draped/*.shp")
shapefiles = [x for x in shapefiles if "OSBS" in x]
np.random.shuffle(shapefiles)
rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
HSI_pool = glob.glob(config["HSI_sensor_pool"], recursive=True)

futures = []
with ThreadPoolExecutor(max_workers=16) as ex:
    for i in shapefiles:
        shp = gpd.read_file(i)
        basename = os.path.splitext(os.path.basename(i))[0]
        try:
            shp = shp.sample(n=1000)
        except Exception:
            continue
        hsi_path = neon_paths.lookup_and_convert(
            bounds=shp.total_bounds,
            rgb_pool=rgb_pool,
            hyperspectral_pool=HSI_pool,
            savedir=config["HSI_tif_dir"],
        )
        for index, row in shp.iterrows():
            futures.append(
                ex.submit(
                    patches.crop,
                    bounds=row["geometry"].bounds,
                    sensor_path=hsi_path,
                    savedir="/orange/idtrees-collab/mining/",
                    basename="{}_{}".format(basename, index),
                )
            )
    for fut in as_completed(futures):
        fut.result()

def remove(x):
    i = rio.open(x).read()
    if not np.isfinite(i).all():
        os.remove(x)

#Make sure all data is valid.
images = glob.glob("/orange/idtrees-collab/mining/*.tif")
with ThreadPoolExecutor(max_workers=16) as ex:
    remove_futures = [ex.submit(remove, x) for x in images]
    for fut in as_completed(remove_futures):
        fut.result()

images = glob.glob("/orange/idtrees-collab/mining/*.tif")
mining = pd.DataFrame({"image_path":images})
mining.to_csv("/orange/idtrees-collab/mining/mining.csv")
