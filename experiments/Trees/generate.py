#Generate tfrecords
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils.start_cluster import start
from DeepTreeAttention.utils.paths import lookup_and_convert

from distributed import wait
import glob
import os

att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")

#get root dir full path
client = start(cpus=3, mem_size="12GB") 

#Generate training data
train_tfrecords = []
weak_records = glob.glob(os.path.join("/orange/idtrees-collab/species_classification/confident_predictions","*.csv"))
weak_records = ["BART" in x for x in weak_records]
weak_records = weak_records[:3]

for record in weak_records:
    sensor_path = lookup_and_convert(shapefile, rgb_pool=att.config["train"]["rgb_sensor_pool"], hyperspectral_pool=att.config["train"]["hyperspectral_sensor_pool"], savedir=att.config["hyperspectral_tif_dir"])
    future = client.submit(att.generate, record=record, sensor_path=sensor_path, chunk_size=500, train=True)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    print(x.result())
        
