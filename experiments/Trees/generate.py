#Generate tfrecords
import glob
import numpy as np
import os
import pandas as pd

from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils.start_cluster import start
from DeepTreeAttention.utils.paths import *

from distributed import wait

#get root dir full path
client = start(cpus=3, mem_size="12GB") 

#Generate training data
weak_records = glob.glob(os.path.join("/orange/idtrees-collab/species_classification/confident_predictions","*.csv"))
weak_records = [x for x in weak_records if "BART" in x]
weak_records = weak_records[:3]

print("Running records: {}".format(weak_records))

def run(record):
    
    att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    site_classes_file = "{}/data/processed/site_class_labels.csv".format(ROOT)
    species_classes_file = "{}/data/processed/species_class_labels.csv".format(ROOT)
    
    rgb_pool = glob.glob(rgb_glob, recursive=True)
    hyperspectral_pool = glob.glob(hyperspectral_glob, recursive=True)
    
    #Convert h5 hyperspec
    renamed_record = record.replace("itc_predictions", "image")
    h5_future = lookup_and_convert(shapefile=renamed_record, rgb_pool=rgb_pool, hyperspectral_pool=hyperspectral_pool, savedir=att.config["hyperspectral_tif_dir"])
    wait(h5_future)
    
    rgb_path = find_sensor_path(shapefile=renamed_record, lookup_pool=rgb_pool)
    
    #infer site
    site = site_from_path(renamed_record)
    numeric_site = site_label_dict[site] 
    
    #infer elevation
    h5_path = find_sensor_path(shapefile=renamed_record, lookup_pool=hyperspectral_pool)    
    elevation = elevation_from_tile(h5_path)
    
    #Generate record when complete   
    df = pd.read_csv(record)
    heights = np.repeat(10, df.shape[0])
    
    tfrecords = att.generate(
        csv_file=record,
        HSI_sensor_path=h5_future.result(),
        RGB_sensor_path =rgb_path,
        chunk_size=500,
        train=True,
        site=numeric_site,
        heights=heights,
        elevation=elevation,
        site_classes_file=site_classes_file,
        species_classes_file=species_classes_file)
    
    return tfrecords
    
train_tfrecords = []
for record in weak_records:
    future = client.submit(run, record=record)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    print(x.result())
        
