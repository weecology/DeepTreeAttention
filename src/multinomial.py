#Plot abundance distribution
import numpy as np
import pandas as pd
import glob
import geopandas as gpd
import traceback

from distributed import wait

def run(tile, confusion_path="data/processed/confusion_matrix.csv", iteration=0):
    """Load a shapefile and confusion .csv and sample the confidence probabilities"""
    predicted_tile = gpd.read_file(tile)

    #Load confusion matrix
    confusion = load_confusion(confusion_path)
    
    #Create label dictionary and assign sampled label
    label_dict = {key: value for key, value in enumerate(confusion.keys())}

    # Add dead class
    label_dict["DEAD"] = "DEAD"
    
    predicted_tile["sampled_label"] = predicted_tile.ensembleTa.apply(lambda x: sample_confusion(x, confusion))
    predicted_tile["sampled_taxonID"] = predicted_tile.sampled_label.apply(lambda x: label_dict[x])
    
    tile_count = predicted_tile.sampled_taxonID.value_counts()
    
    return tile_count

def format_confusion_json(path):
    """Starting from the .json objet produced by comet"""
    j = pd.read_json(path)
    df = pd.DataFrame(j.matrix)
    df = df.matrix.apply(lambda x: [y/sum(x) for y in x])
    df = df.apply(pd.Series)    
    df.columns = j.labels
    df["predicted"] = j.labels
    
    return df
    
def load_confusion(path):
    df = pd.read_csv(path, index_col=0)
    flat_dict = {}
    df = df.set_index("predicted")
    for taxonID in df:
        flat_dict[taxonID] = df.loc[taxonID].values
    
    return flat_dict
    
def sample_confusion(taxonID, confusion):
    if taxonID == "DEAD":
        return "DEAD"
    else:
        scores = confusion[taxonID]
        scores = np.array([float(x) for x in scores])
        random_draw = np.random.multinomial(1, scores)
        
        return np.argmax(random_draw)

def wrapper(client, iteration, experiment_key, shp_dir="/blue/ewhite/b.weinstein/DeepTreeAttention/results/", savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results"):  
    tiles = glob.glob("{}/{}/*.shp".format(shp_dir, experiment_key))
    total_counts = pd.Series()
    counts = []
    for tile in tiles:
        future = client.submit(run, tile=tile, iteration=iteration)
        counts.append(future)
    
    wait(counts)
    
    for result in counts:
        try:
            ser = result.result()
        except Exception as e:
            traceback.print_exc(e)
            print(e)
            continue
        total_counts = total_counts.add(ser, fill_value=0)
    total_counts.sort_values()
    total_counts.to_csv("{}/{}/multinomial_permutation_{}.csv".format(savedir, experiment_key, iteration))