#Plot abundance distribution
from glob import glob
import os
import pandas as pd
import geopandas as gpd
from src import start_cluster

client = start_cluster.start(cpus=30,mem_size="5GB")

#Same data

#species_model_paths = #"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt",
                       #"/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ac7b4194811c4bdd9291892bccc4e661.pt",
species_model_paths = ["/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/0df3483cc211460caefb73b6fd369c4b.pt"]

def read_shp(path):
    gdf = gpd.read_file(path)
    gdf = gdf.groupby("individual").apply(lambda x: x.head(1))    
    #limit by OSBS polygon
    #boundary = gpd.read_file("/home/b.weinstein/DeepTreeAttention/data/raw/OSBSBoundary/OSBS_boundary.shp")
    #boundary = boundary.to_crs("epsg:32617")
    #intersects = gpd.clip(gdf, boundary)
    
    return gdf

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
    shps = [x.result() for x in futures]
    combined_shps = pd.concat(shps)
    gpd_boundary = gpd.GeoDataFrame(combined_shps, geometry="geometry")
    gpd_boundary = gpd_boundary.reset_index(drop=True)
    gpd_boundary.to_file("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/JERC_predictions.shp".format(basename))
