#Plot abundance distribution
import numpy as np
import pandas as pd
import glob
import geopandas as gpd

from distributed import wait

def run(tile, confusion_path):
    """Load a shapefile and confusion .csv and sample the confidence probabilities"""
    predicted_tile = gpd.read_file(tile)

    #Load confusion matrix
    confusion = load_confusion(confusion_path)
    
    #Create label dictionary and assign sampled label
    label_dict = {}
    for name, group in predicted_tile.groupby("ensemble_l"):
        label_dict[name] = group.ensembleTa.unique()[0]
        
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
    df = pd.read_csv(path)
    as_dict = df.set_index("predicted").to_dict()
    flat_dict = {}
    for key, value in as_dict.items():
        flat_dict[key] = [j for i,j in value.items()]
    
    return flat_dict
    
def sample_confusion(taxonID, confusion):
    if taxonID == "DEAD":
        return "DEAD"
    else:
        scores = confusion[taxonID]
        scores = np.array([float(x) for x in scores])
        random_draw = np.random.multinomial(1, scores)
        
        return np.argmax(random_draw)

def wrapper(client, iteration, experiment_key, savedir="/blue/ewhite/b.weinstein/DeepTreeAttention/results"):  
    tiles = glob.glob("{}/{}/*.shp".format(savedir, experiment_key))
    total_counts = pd.Series()
    counts = []
    for tile in tiles:
        future = client.submit(run, tile=tile)
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
    total_counts.to_csv("{}/{}/multinomial_permutation_{}.csv".format(savedir, experiment_key, iteration))