#Plot abundance distribution
from distributed import wait
import numpy as np
import pandas as pd
from src.data import read_config
import os
import glob
from functools import reduce
import geopandas as gpd

def run(tile, dirname="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/"):
    config = read_config("config.yml")
    predicted_tile = gpd.read_file("{}/{}.shp".format(dirname, tile))
    level0 =  pd.read_csv(os.path.join(dirname, "{}_0.csv".format(tile)), index_col=0)
    level1 =  pd.read_csv(os.path.join(dirname, "{}_1.csv".format(tile)), index_col=0)
    level2 =  pd.read_csv(os.path.join(dirname, "{}_2.csv".format(tile)), index_col=0)
    level3 =  pd.read_csv(os.path.join(dirname, "{}_3.csv".format(tile)), index_col=0)
    level4 =  pd.read_csv(os.path.join(dirname, "{}_4.csv".format(tile)), index_col=0)
    
    ##Multiple each species by the confidence scores propogated by each level     
    ensembleTaxonID = []
    for i in range(level0.shape[0]):        
        PIPA = level0["0"][i]
        NYSY = level0["1"][i] * level1["1"][i] * level2["0"][i]
        ACRU = level0["1"][i] * level1["1"][i] * level2["1"][i]
        CAGL8 = level0["1"][i] * level1["1"][i] * level2["2"][i]
        MAGNO = level0["1"][i] * level1["1"][i] * level2["3"][i]
        LIST2 = level0["1"][i] * level1["1"][i] * level2["4"][i]
        PICL = level0["1"][i] * level1["0"][i] * level3["0"][i]
        PIEL = level0["1"][i] * level1["0"][i] * level3["0"][i]
        PITA = level0["1"][i] * level1["0"][i] * level3["0"][i]
        QULA2 = level0["1"][i] * level1["1"][i] * level2["5"][i] * level4["0"][i]
        QUGE2 = level0["1"][i] * level1["1"][i] * level2["5"][i] * level4["1"][i]
        QUHE2 = level0["1"][i] * level1["1"][i] * level2["5"][i] * level4["2"][i]
        QUNI = level0["1"][i] * level1["1"][i] * level2["5"][i] * level4["3"][i]
        QUVI = level0["1"][i] * level1["1"][i] * level2["5"][i] * level4["4"][i]
        prob_dict = {"PIPA":PIPA,"NYSY":NYSY,"ACRU":ACRU,"CAGL8":CAGL8,"MAGNO":MAGNO,
         "LIST2":LIST2,"PICL":PICL,"PIEL":PIEL,"PITA":PITA,"QULA2":QULA2,"QUGE2":QUGE2,"QUHE2":QUHE2,"QUNI":QUNI,"QUVI":QUVI}
        probs = [value for key, value in prob_dict.items()]
        labels = [key for key, value in prob_dict.items()]
        index = np.argmax(np.random.multinomial(1, probs/np.sum(probs)))
        ensembleTaxonID.append(labels[index])
    
    #level_results = []
    #for level, df in enumerate(levels):
        #level_results.append(format_level(df=df, level=level, label_to_taxonIDs=m.label_to_taxonIDs[level]))
    
    #results = reduce(lambda  left,right: pd.merge(left,right,on=['individual'],
                                                    #how='outer'), level_results) 
    
    #trees = ensemble(results, m.species_label_dict)
    
    # Remove predictions for dead trees
    trees = pd.DataFrame({"individual":level0.individual,"ensembleTaxonID":ensembleTaxonID})
    trees = trees.merge(predicted_tile[["individual","dead_label","dead_score"]], on="individual")    
    trees.loc[(trees.dead_label==1) & (trees.dead_score > config["dead_threshold"]),"ensembleTaxonID"] = "DEAD"
    tile_count = trees.ensembleTaxonID.value_counts()
    
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
    a = df.drop(columns=["individual"]).values
    #Sanitize prob vectors to sum to 1, rounding errors from .csv
    pred_label_top1 = []
            
    # Sample multinomial, round rare overflow errors
    for row in a:
        random_draw = np.random.multinomial(1,row/np.sum(row)) 
        pred_label_top1.append(np.argmax(random_draw))
    
    top1_score = a[np.arange(len(a)),pred_label_top1]
    results = pd.DataFrame({
        "pred_label_top1_level_{}".format(level): pred_label_top1,
        "top1_score_level_{}".format(level): top1_score,
        "individual": df["individual"]        
    })
    results["pred_taxa_top1_level_{}".format(level)] = results["pred_label_top1_level_{}".format(level)].apply(lambda x: label_to_taxonIDs[x]) 
    
    return results

def wrapper(client, iteration, savedir):  
    files = glob.glob("/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/*.csv")
    files = [x for x in files if "OSBS" in x]
    tiles = np.unique(["_".join(os.path.splitext(os.path.basename(x))[0].split("_")[:-1]) for x in files])
    total_counts = pd.Series()
    counts = []
    for tile in tiles:
        future = client.submit(run, tile=tile, dirname="/blue/ewhite/b.weinstein/DeepTreeAttention/results/06ee8e987b014a4d9b6b824ad6d28d83/")
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