#Plot abundance distribution
from distributed import wait
import numpy as np
import pandas as pd
import os
import glob
import geopandas as gpd

def run(tile, dirname="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/",iteration=0):

    #Create label dictionary
    label_dict = predicted_tile.groupby("ensemble_label").ensembleTa.head(1).to_dict()
    
    predicted_tile = gpd.read_file("{}/{}.shp".format(dirname, tile))
    # Remove predictions for dead trees
    
    #Load confusion matrix
    confusion = load_confusion(path)
    predicted_tile["sampled_label"] = predicted_tile.ensembleTa.apply(lambda x: sample_confusion(confusion.loc[x]))
    predicted_tile["sampled_taxonID"] = predicted_tile.sampled_label.apply(lambda x: label_dict[x])
    
    tile_count = predicted_tile.sampled_taxonID.value_counts()
    
    return tile_count

def load_confusion(path):
    j = pd.read_json("/Users/benweinstein/Downloads/confusion-matrix.json")
    df = pd.DataFrame(j.matrix)
    df = df.matrix.apply(lambda x: [y/sum(x) for y in x])
    df.columns = j.labels
    df["predicted"] = j.labels
    df.set_index("predicted").to_dict()
    
def sample_confusion(scores):
    random_draw = np.random.multinomial(1, scores)
    return np.argmax(random_draw)

def wrapper(client, iteration, savedir):  
    files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/*.csv")
    files = [x for x in files if "OSBS" in x]
    tiles = np.unique(["_".join(os.path.splitext(os.path.basename(x))[0].split("_")[:-1]) for x in files])
    total_counts = pd.Series()
    counts = []
    for tile in tiles:
        future = client.submit(run, tile=tile, dirname="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/", iteration=iteration)
        counts.append(future)
    
    wait(counts)
    
    for result in counts:
        try:
            ser = result.result()
        except Exception as e:
            print(e)
            continue
        total_counts = total_counts.add(ser, fill_value=0)
    total_counts.sort_values()
    total_counts.to_csv("{}/abundance_permutation_{}.csv".format(savedir, iteration))