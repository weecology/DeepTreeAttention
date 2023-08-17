import glob
import os
import geopandas as gpd
import sys
print(sys.path)
from src.start_cluster import start

def check_tile_year(x):
    shp = gpd.read_file(x)
    if shp.tile_year.unique()[0] == "image":
        os.remove(x)
        return(x)

client = start(cpus=150,mem_size="6GB")
client.proccess=False
files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/site_crops/TALL/shp/*.shp", recursive=True)
futures = client.map(check_tile_year,files)

for future in futures:
    print(future.result())