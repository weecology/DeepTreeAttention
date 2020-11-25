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

att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")

#get root dir full path
client = start(cpus=3, mem_size="12GB") 

#Generate training data
weak_records = glob.glob(os.path.join("/orange/idtrees-collab/species_classification/confident_predictions","*.csv"))
weak_records = [x for x in weak_records if "BART" in x]
weak_records = weak_records[:3]

print("Running records: {}".format(weak_records))

rgb_pool = glob.glob(att.config["rgb_sensor_pool"],recursive=True)
hyperspectral_pool = glob.glob(att.config["hyperspectral_sensor_pool"],recursive=True)

train_tfrecords = []
for record in weak_records:
    #Hot fix for the regex, sergio changed the name slightly.
    
    #Convert h5 hyperspec
    renamed_record = record.replace("itc_predictions", "image")
    h5_future = client.submit(lookup_and_convert,shapefile=renamed_record,rgb_pool=rgb_pool, hyperspectral_pool=hyperspectral_pool, savedir=att.config["hyperspectral_tif_dir"])
    wait(h5_future)
    
    rgb_path = find_sensor_path(shapefile=renamed_record, lookup_pool=rgb_pool)
    
    #infer site
    site = site_from_path(renamed_record)
    
    #infer elevation
    h5_path = find_sensor_path(shapefile=renamed_record, lookup_pool=hyperspectral_pool)    
    elevation = elevation_from_tile(h5_path)
    
    #Generate record when complete
    
    #TODO fix heights, hardcode to bypass while testing
    df = pd.read_csv(record)
    heights = np.repeat(10,df.shape[0])
    
    future = client.submit(att.generate, shapefile=renamed_record, HSI_sensor_path=h5_future.result(), RGB_sensor_path =rgb_path , chunk_size=500, train=True, site=site, heights =heights , elevation=elevation)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    print(x.result())
        
