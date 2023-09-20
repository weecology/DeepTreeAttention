from src import start_cluster
from src.model_list import species_model_paths
import geopandas
import pandas as pd
from dask import delayed
import dask.dataframe as dd
import glob
import os

client = start_cluster.start(cpus=120, mem_size="5GB")

def read_shp(path):
    gdf = geopandas.read_file(path)
    df = pd.DataFrame(gdf)
    
    return df

shps = []
for site in species_model_paths:
    x = species_model_paths[site]
    basename = os.path.splitext(os.path.basename(x))[0]
    site_predictions = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, basename), recursive=True)   
    shps.append(site_predictions)
    
dfs = client.map(read_shp, shps)
ddf = dd.from_delayed(dfs)

