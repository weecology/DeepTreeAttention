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
    df = pd.DataFrame(gdf[["sci_name","ens_score","site_id"]])
    
    return df

# remove JERC key
species_model_paths = {key:value for key, value in species_model_paths.items() if key != "JERC"}
species_model_paths["STEI"] = species_model_paths["TREE"]

shps = []
for site in species_model_paths:
    x = species_model_paths[site]
    basename = os.path.splitext(os.path.basename(x))[0]
    if site == "STEI":
        site = "TREE"
    site_predictions = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/{}/*.shp".format(site, basename), recursive=True)   
    shps.append(site_predictions)
    
# flatten list
shps = [item for sublist in shps for item in sublist]

# as a test
a = read_shp(shps[0])

dfs = client.map(read_shp, shps)
ddf = dd.from_delayed(dfs)
ddf.shape[0].compute()
ddf.site_id.unique().compute().shape[0]
site_counts = ddf.site_id.value_counts().compute()

ddf.site_id.value_counts().mean()
