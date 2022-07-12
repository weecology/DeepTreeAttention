#Plot abundance distribution
import dask
import dask.dataframe as dd
import distributed
import sys
import numpy as np
import pandas as pd
sys.path.append("/home/b.weinstein/DeepTreeAttention")
from src import start_cluster
from src.data import read_config
from src.models import multi_stage
import os
import glob
from functools import reduce

#client = start_cluster.start(cpus=2)
#client = distributed.Client()

def run(tile, dirname):
    #Load model to get label dicts
    config = read_config("../config.yml")
    species_model_path = "/blue/ewhite/b.weinstein/DeepTreeAttention/snapshots/06ee8e987b014a4d9b6b824ad6d28d83.pt"
    m = multi_stage.MultiStage.load_from_checkpoint(species_model_path, config=config)
    
    level0 =  pd.read_csv(os.path.join(dirname, "{}_0.csv".format(tile)))
    level1 =  pd.read_csv(os.path.join(dirname, "{}_1.csv".format(tile)))
    level2 =  pd.read_csv(os.path.join(dirname, "{}_2.csv".format(tile)))
    level3 =  pd.read_csv(os.path.join(dirname, "{}_3.csv".format(tile)))
    level4 =  pd.read_csv(os.path.join(dirname, "{}_4.csv".format(tile)))
    
    levels = [level0, level1, level2, level3, level4]
    levels = [x.drop(columns="Unnamed: 0") for x in levels]
    
    level_results = []
    for level, df in enumerate(levels):
        level_results.append(format_level(df=df, level=level, label_to_taxonIDs=m.label_to_taxonIDs[level]))
    
    results = reduce(lambda  left,right: pd.merge(left,right,on=['individual'],
                                                    how='outer'), level_results) 
    
    ensemble_df = ensemble(results, m.species_label_dict)
    tile_count = ensemble_df.ensembleTa.value_counts()
    
    return tile_count

def ensemble(results, species_label_dict):
    """Given a multi-level model, create a final output prediction and score"""
    ensemble_taxonID = []
    ensemble_label = []
    ensemble_score = []
    
    for index,row in results.iterrows():
        try:    
            if row["pred_taxa_top1_level_0"] == "PIPA2":
                ensemble_taxonID.append("PIPA2")
                ensemble_label.append(species_label_dict["PIPA2"])
                ensemble_score.append(row["top1_score_level_0"])                
            else:
                if row["pred_taxa_top1_level_1"] == "BROADLEAF":
                    if row["pred_taxa_top1_level_2"] == "OAK":
                        ensemble_taxonID.append(row["pred_taxa_top1_level_4"])
                        ensemble_label.append(species_label_dict[row["pred_taxa_top1_level_4"]])
                        ensemble_score.append(row["top1_score_level_4"])
                    else:
                        ensemble_taxonID.append(row["pred_taxa_top1_level_2"])
                        ensemble_label.append(species_label_dict[row["pred_taxa_top1_level_2"]])
                        ensemble_score.append(row["top1_score_level_2"])                     
                else:
                    ensemble_taxonID.append(row["pred_taxa_top1_level_3"])
                    ensemble_label.append(species_label_dict[row["pred_taxa_top1_level_3"]])
                    ensemble_score.append(row["top1_score_level_3"])
        except Exception as e:
            print("row {} failed with {}".format(row, e))            
    
    results["ensembleTaxonID"] = ensemble_taxonID
    results["ens_score"] = ensemble_score
    results["ens_label"] = ensemble_label   
    
    return results

def format_level(df, level, label_to_taxonIDs):
    #Loop through each list and get multonial draw
    probs = df.drop(columns=["individual"])
    pred_label_top1 = np.argmax(np.vstack(probs.apply(lambda x: np.random.multinomial(1,x), axis=1).values),1)
    top1_score = np.max(np.vstack(probs.apply(lambda x: np.random.multinomial(1,x), axis=1).values),1)
    results = pd.DataFrame({
        "pred_label_top1_level_{}".format(level):pred_label_top1,
        "top1_score_level_{}".format(level):top1_score,
        "individual":df["individual"]        
    })
    results["pred_taxa_top1_level_{}".format(level)] = results["pred_label_top1_level_{}".format(level)].apply(lambda x: label_to_taxonIDs[x]) 
    
    return results

files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/*.csv")
tiles = np.unique(["_".join(os.path.splitext(os.path.basename(x))[0].split("_")[:-1]) for x in files])
for tile in tiles:
    counts = run(tile, "/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/")

total_counts = pd.Series()
for ser in counts:
    total_counts = total_counts.add(ser, fill_value=0)

total_counts.sort_values()
total_counts.sum()