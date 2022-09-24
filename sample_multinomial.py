from src.multinomial import *
from src import start_cluster
import glob
import pandas as pd

#client = start_cluster.start(cpus=50, mem_size="5GB")
#for x in range(2):
    #wrapper(client=client, iteration=x, experiment_key="209ca047ed004d778c0f0e728e126bda")

for x in range(100):  
    tiles = glob.glob("{}/{}/*.shp".format("/blue/ewhite/b.weinstein/DeepTreeAttention/results/", "06ee8e987b014a4d9b6b824ad6d28d83"))
    total_counts = pd.Series()
    counts = []
    for tile in tiles:
        count = run(tile)
        counts.append(count)
    
    for count in counts:
        total_counts = total_counts.add(count, fill_value=0)
    total_counts.sort_values()
    total_counts.to_csv("/blue/ewhite/b.weinstein/DeepTreeAttention/results/{}/multinomial_permutation_{}.csv".format(experiment_key, x))