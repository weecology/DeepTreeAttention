#Generate tfrecords
import glob
import numpy as np
import os
import pandas as pd
from dask import dataframe as dd

from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils.start_cluster import start
from DeepTreeAttention.utils.paths import *

from distributed import wait

#Delete any file previous run
old_files = glob.glob("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/*")
[os.remove(x) for x in old_files]
old_files = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/pretraining/*")
[os.remove(x) for x in old_files]


#get root dir full path
client = start(cpus=10, mem_size="15GB") 

weak_records = glob.glob(os.path.join("/orange/idtrees-collab/species_classification/confident_predictions","*.csv"))

#Check if complete
def check_shape(x):
    df = pd.read_csv(x)
    if len(df.columns) == 12:
        return x
    else:
        return None
    
futures = client.map(check_shape,weak_records)
completed_records = [x.result() for x in futures if x.result() is not None]

#Create a dask dataframe of csv files
df = dd.read_csv(completed_records[0:100], include_path_column = True)

#Get a balanced set of species
df = df.groupby("filtered_taxonID").apply(lambda x: x.reset_index().head(2000)).compute()

#write a csv file per tile
def write_csv(x):
    if x.empty:
        return None
    path_name = x.path.unique()[0]
    basename = os.path.basename(path_name)
    x.to_csv("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/{}".format(basename))

df.groupby("path").apply(write_csv)

#Generate training data
records_to_run = glob.glob("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/*.csv")
print("Running records: {}".format(records_to_run))

def run(record):
    
    att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    site_classes_file = "{}/data/processed/site_class_labels.csv".format(ROOT)
    site_classdf  = pd.read_csv(site_classes_file)
    site_label_dict = site_classdf.set_index("siteID").label.to_dict()    
    
    species_classes_file = "{}/data/processed/species_class_labels.csv".format(ROOT)
    species_classdf  = pd.read_csv(species_classes_file)
    species_label_dict = species_classdf.set_index("taxonID").label.to_dict()
    
    rgb_pool = glob.glob(att.config["rgb_sensor_pool"], recursive=True)
    hyperspectral_pool = glob.glob(att.config["hyperspectral_sensor_pool"], recursive=True)
    
    #Convert h5 hyperspec
    renamed_record = record.replace("itc_predictions", "image")
    hyperspec_path = lookup_and_convert(shapefile=renamed_record, rgb_pool=rgb_pool, hyperspectral_pool=hyperspectral_pool, savedir=att.config["hyperspectral_tif_dir"])
    
    rgb_path = find_sensor_path(shapefile=renamed_record, lookup_pool=rgb_pool)
    
    #infer site
    site = site_from_path(renamed_record)
    numeric_site = site_label_dict[site] 
    
    #infer elevation
    h5_path = find_sensor_path(shapefile=renamed_record, lookup_pool=hyperspectral_pool)    
    elevation = elevation_from_tile(h5_path)
    
    #Generate record when complete   
    df = pd.read_csv(record)
    
    # hot fix the heights for the moment.
    heights = np.repeat(10, df.shape[0])
    
    tfrecords = att.generate(
        csv_file=record,
        HSI_sensor_path=hyperspec_path,
        RGB_sensor_path=rgb_path,
        chunk_size=500,
        train=True,
        site=numeric_site,
        heights=heights,
        elevation=elevation,
        label_column="filtered_taxonID",
        species_label_dict=species_label_dict)
    
    return tfrecords
    
train_tfrecords = []
for record in records_to_run:
    future = client.submit(run, record=record)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    try:
        print(x.result())
    except:
        pass
        
