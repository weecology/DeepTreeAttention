#Generate tfrecords
import glob
import os
import pandas as pd
import geopandas as gpd
import tensorflow as tf

from DeepTreeAttention.trees import AttentionModel, __file__
from DeepTreeAttention.utils.start_cluster import start
from DeepTreeAttention.utils.paths import *
from distributed import wait

#Delete any file previous run
old_files = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*")
[os.remove(x) for x in old_files]
old_files = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*")
[os.remove(x) for x in old_files]

#get root dir full path
client = start(cpus=60, mem_size="10GB") 

def run(record, savedir):
    """Take a plot of deepforest prediction (see prepare_field_data.py) and generate crops for training/evalution"""
    #Read record
    df = gpd.read_file(record)
    
    att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    site_classes_file = "{}/data/processed/site_class_labels.csv".format(ROOT)
    site_classdf  = pd.read_csv(site_classes_file)
    site_label_dict = site_classdf.set_index("siteID").label.to_dict()    
    
    domain_classes_file = "{}/data/processed/domain_class_labels.csv".format(ROOT)
    domain_classdf  = pd.read_csv(domain_classes_file)
    domain_label_dict = domain_classdf.set_index("domainID").label.to_dict()  
    
    species_classes_file = "{}/data/processed/species_class_labels.csv".format(ROOT)
    species_classdf  = pd.read_csv(species_classes_file)
    species_label_dict = species_classdf.set_index("taxonID").label.to_dict()
    
    rgb_pool = glob.glob(att.config["rgb_sensor_pool"], recursive=True)
    hyperspectral_pool = glob.glob(att.config["hyperspectral_sensor_pool"], recursive=True)
    
    #Convert h5 hyperspec
    hyperspec_path = lookup_and_convert(shapefile=record, rgb_pool=rgb_pool, hyperspectral_pool=hyperspectral_pool, savedir=att.config["hyperspectral_tif_dir"])
    rgb_path = find_sensor_path(shapefile=record, lookup_pool=rgb_pool)
    
    #infer site, only 1 per plot.
    site = df.siteID.unique()[0]
    numeric_site = site_label_dict[site] 
    
    domain = df.domainID.unique()[0]
    numeric_domain = domain_label_dict[domain] 
    
    #infer elevation
    h5_path = find_sensor_path(shapefile=record, lookup_pool=hyperspectral_pool)    
    elevation = elevation_from_tile(h5_path)
    
    ensemble_model = tf.keras.Model(ensemble_model.inputs, ensemble_model.get_layer("submodel_concat").output)
    
    #Generate record when complete   
    tfrecords = att.generate(
        csv_file=record,
        HSI_sensor_path=hyperspec_path,
        RGB_sensor_path=rgb_path,
        chunk_size=500,
        train=True,
        domain=numeric_domain,
        site=numeric_site,
        elevation=elevation,
        label_column="filtered_taxonID",
        species_label_dict=species_label_dict,
        ensemble_model=ensemble_model,
        savedir=savedir
    )
    
    return tfrecords

#test
plots_to_run = glob.glob("data/deepforest_boxes/evaluation/*.shp")

test_tfrecords = []
for record in plots_to_run:
    future = client.submit(run, record=record,  savedir="/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/")
    train_tfrecords.append(future)
    
wait(test_tfrecords)
for x in test_tfrecords:
    try:
        print(x.result())
    except Exception as e:
        print("{} failed with {}".format(x, e))
        pass

#train
plots_to_run = glob.glob("data/deepforest_boxes/train/*.shp")
train_tfrecords = []
for record in plots_to_run:
    future = client.submit(run, record=record, savedir="/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/")
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    try:
        print(x.result())
    except Exception as e:
        print("{} failed with {}".format(x, e))
        pass
      
