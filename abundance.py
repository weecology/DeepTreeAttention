#Plot abundance distribution
from glob import glob
import os
import pandas as pd
import geopandas as gpd
from src import start_cluster

client = start_cluster.start(cpus=150,mem_size="10GB")

##Same data

species_model_paths = ["/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0df3483cc211460caefb73b6fd369c4b.pt"]

def read_shp(path):
    gdf = gpd.read_file(path)
    #limit by OSBS polygon
    #boundary = gpd.read_file("/home/b.weinstein/DeepTreeAttention/data/raw/OSBSBoundary/OSBS_boundary.shp")
    #One individual per time slice
    gdf = gdf.groupby("individual").apply(lambda x: x.head(1))
    
    #boundary = boundary.to_crs("epsg:32617")
    #intersects = gpd.clip(gdf, boundary)
    tile_count = gdf.ensembleTa.value_counts()
    
    return tile_count

futures = []
for species_model_path in species_model_paths:
    print(species_model_path)
    basename = os.path.splitext(os.path.basename(species_model_path))[0]
    input_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/*_image.shp".format(basename)
    files = glob(input_dir)
    print(files)
    if len(files) == 0:
        continue
    counts = []
    futures = client.map(read_shp,files)
    counts = [x.result() for x in futures]
    total_counts = pd.Series()
    for ser in counts:
        total_counts = total_counts.add(ser, fill_value=0)
    total_counts.sort_values()
    total_counts.sum()
    total_counts.to_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/abundance.csv".format(basename))
    
all_abundance = []
for species_model_path in species_model_paths:
    basename = os.path.splitext(os.path.basename(species_model_path))[0]    
    try:
        df = pd.read_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/abundance.csv".format(basename))
    except:
        continue
    df["path"] = basename
    all_abundance.append(df)

all_abundance = pd.concat(all_abundance)
all_abundance.columns = ["taxonID","count","model"]
all_abundance.to_csv("results/jerc_abundance.csv")

