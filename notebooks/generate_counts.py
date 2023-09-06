import geopandas as gpd
import glob
import os
import pandas as pd
from src import start_cluster
from src.model_list import species_model_paths

client = start_cluster.start(cpus=120, mem_size="5GB")
def read_file(f):
    return gpd.read_file(f)[["sci_name"]].value_counts().reset_index() 

for site in species_model_paths:
    print(site)
    model = species_model_paths[site]
    prediction_dir = os.path.join("/blue/ewhite/b.weinstein/DeepTreeAttention/results/predictions/{}/".format(site),
                                      os.path.splitext(os.path.basename(model))[0])  
    files = glob.glob("{}/*.shp".format(prediction_dir))
    if len(files) == 0:
        continue
    total_counts = []

    futures = client.map(read_file, files)
    for f in futures:
        try:
            ser = f.result()
        except:
            continue    
        total_counts.append(ser)
    total_counts = pd.concat(total_counts).groupby("sci_name").sum()
    total_counts.to_csv("/home/b.weinstein/DeepTreeAttention/results/{}_abundance.csv".format(site))
    
        