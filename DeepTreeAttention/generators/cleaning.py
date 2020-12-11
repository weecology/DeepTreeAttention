#Cleanlab
from DeepTreeAttention import trees
from cleanlab.pruning import get_noise_indices
import numpy as np
import pandas as pd
import geopandas as gpd

def clean_labels():     
    att= trees.AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    att.create()
    att.ensemble_model.load_weights("{}/Ensemble_model.h5".format(att.config["train"]["checkpoint_dir"]))
    att.read_data(mode="ensemble")
    
    y_true = [ ]
    y_pred = [ ]
    box_index = [ ]
    
    for index, batch in att.val_split_with_ids:
        data,label = batch
        prediction = att.ensemble_model.predict_on_batch(data)            
        y_true.append(label)
        y_pred.append(prediction)
        box_index.append(index)            
        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    box_index = np.concatenate(box_index)
    box_index = list(box_index)
    
    y_true = np.argmax(y_true, 1)
    y_pred = np.argmax(y_pred, 1)
        
    results = pd.DataFrame({"true":y_true,"predicted":y_pred, "box_index":box_index})
    results["id"] = results["box_index"].apply(lambda x: int(x.decode("utf-8")))
    
    #Read original data        
    shapefile = att.config["evaluation"]["ground_truth_path"]
    gdf = gpd.read_file(shapefile)        

    #Merge
    joined_gdf = gdf.merge(results, on="id")
    joined_gdf = joined_gdf.drop(columns=["box_index"])
    
    labeldf = pd.read_csv(att.classes_file)
    label_names = list(labeldf.taxonID.values)
    
    joined_gdf["true_taxonID"] = joined_gdf.true.apply(lambda x: label_names[x])
    joined_gdf["predicted_taxonID"] = joined_gdf.predicted.apply(lambda x: label_names[x])
    
    ordered_label_errors = get_noise_indices(
        s=y_true,
        psx=y_pred
     )
    
    joined_gdf["label_errors"] = ordered_label_errors
    
    return joined_gdf
    
if __name__ == "__main__":
    joined_gdf = clean_labels()
    joined_gdf.head()