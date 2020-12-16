#Cleanlab
from DeepTreeAttention import trees
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import tensorflow.keras as tfk

import cleanlab
from DeepTreeAttention.models.layers import WeightedSum

import numpy as np
import pandas as pd
import geopandas as gpd

def autoencoder_model(height, width, channels):
    input_shape = (height, width, channels)
    sensor_inputs = layers.Input(shape=input_shape, name="data_input")
        
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(sensor_inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
        
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(channels, (3, 3), activation='sigmoid', padding='same', kernel_regularizer=tfk.regularizers.l2())(x)
    
    autoencoder = tfk.Model(sensor_inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    
    return autoencoder
    
def clean_labels():     
    att = trees.AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    att.create()
    att.HSI_model = load_model("/orange/idtrees-collab/DeepTreeAttention/snapshots/20201211_120435/HSI_model.h5", custom_objects={"WeightedSum": WeightedSum}, compile=True)
    att.read_data(mode="HSI")
    
    y_true = [ ]
    y_pred = [ ]
    box_index = [ ]
    
    for index, batch in att.val_split_with_ids:
        data,label = batch
        prediction = att.HSI_model.predict_on_batch(data)            
        y_true.append(label)
        y_pred.append(prediction)
        box_index.append(index)            
        
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    box_index = np.concatenate(box_index)
    box_index = list(box_index)
    
    y_true = np.argmax(y_true, 1)
        
    label_errors_bool = cleanlab.pruning.get_noise_indices(
        s=y_true,
        psx=y_pred,
        sorted_index_method = None
     )
    
    label_errors_idx = cleanlab.pruning.order_label_errors(
        label_errors_bool = label_errors_bool,
        labels=y_true,
        psx=y_pred,
        sorted_index_method = 'normalized_margin',
    )
    
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
    
    joined_gdf["label_errors"] = ordered_label_errors
        
    return joined_gdf
    
if __name__ == "__main__":
    joined_gdf = clean_labels()
    joined_gdf.head()