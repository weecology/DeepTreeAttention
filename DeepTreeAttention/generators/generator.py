"""
Keras sequence generator for DeepTreeAttention. 
The aim of this module is to read in a set of hyperspetral .tif labels for model training or prediction.
Inspired by https://biswajitsahoo1111.github.io/post/reading-multiple-files-in-tensorflow-2/
"""
import tensorflow as tf
import numpy as  np
import pandas as pd

def get_classes(annotation_csv):
    """Create a label dictionary to convert between names and index
        Args:
            annotation_csv: a .csv file with columns "path", "label"
        Returns:
            label_dict: labels -> index
    """
    annotations = pd.read_csv(annotation_csv)
    labels = annotations.label.unique()
    label_dict = {}
    
    for index, label in enumerate(labels):
        label_dict[label] = index
    
    return label
    
def train_generator(annotation_csv, batch_size, shuffle=True):
    """tf.data genertor to iterate through rasters given a set of annotation labels
    Args:
        file_list: a list of .tif raster paths
        annotation_csv: a .csv file with columns "path", "label"
    """
    #Read annotation csv
    annotations = pandas.read_csv(annotation_csv)
    file_list = annotations.path.unique()
    
    #Shuffle on epoch
    i = 0
    while True:

        if i*batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            if shuffle:
                np.random.shuffle(file_list)    
        else:
            data = [ ]
            labels = []
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            
            for f in file_chunk:
                #Read into numpy array
                data = read_raster(f)
                data.append(data)
                
                #gather labels
                image_df = annotations[annotations.path == f]
                label = image_df.label.values[0]
                labels.append(label_dict[label])
            
            #yield as numpy arrays
            data = np.asarray(data).reshape(-1,32,32,1)
            labels = np.asarray(labels)       
            i=+1             
            
            yield data, label
            
        
