#Plot abundance distribution
from glob import glob
import os
import pandas as pd
import geopandas as gpd
from src import start_cluster
import dask
client = start_cluster.start(cpus=5,mem_size="20GB")

species_model_paths = ["/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/ac7b4194811c4bdd9291892bccc4e661.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/b629e5365a104320bcec03843e9dd6fd.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/5ac9afabe3f6402a9c312ba4cee5160a.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/46aff76fe2974b72a5d001c555d7c03a.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/63bdab99d6874f038212ac301439e9cc.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/c871ed25dc1c4a3e97cf3b723cf88bb6.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/6d45510824d6442c987b500a156b77d6.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/83f6ede4f90b44ebac6c1ac271ea0939.pt",
                       "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/47ee5858b1104214be178389c13bd025.pt"
                       ]

def read_shp(path):
    gdf = gpd.read_file(path)
    tile_count = gdf.ensembleTa.value_counts()
    return tile_count
        
futures = []
for species_model_path in species_model_paths:
    print(species_model_path)
    basename = os.path.splitext(os.path.basename(species_model_path))[0]
    input_dir = "/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/*.shp".format(basename)
    files = glob(input_dir)
    print(files)
    counts = []
    futures = dask.delayed([read_shp(x) for x in files[:5]])
    counts = futures.compute()
    print(counts)
    for x in futures:
        counts.append(x.result())
        
    total_counts = pd.Series()
    for ser in counts:
        total_counts = total_counts.add(ser, fill_value=0)
    total_counts.sort_values()
    total_counts.sum()
    total_counts.to_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/abundance.csv".format(basename))