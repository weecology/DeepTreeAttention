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
weak_records = glob.glob(os.path.join("/orange/idtrees-collab/species_classification/confident_predictions","*.csv"))
weak_records = [x for x in weak_records if "BART" in x]
weak_records = weak_records[:3]

print("Running records: {}".format(weak_records))

rgb_pool = glob.glob(att.config["rgb_sensor_pool"])
hyperspectral_pool = glob.glob(att.config["hyperspectral_sensor_pool"])

for record in weak_records:
    #Hot fix for the regex, sergio changed the name slightly.
    record = record.replace("itc_predictions", "image")
    sensor_path = lookup_and_convert(shapefile=record, rgb_pool=rgb_pool, hyperspectral_pool=hyperspectral_pool, savedir=att.config["hyperspectral_tif_dir"])
    future = client.submit(att.generate, record=record, sensor_path=sensor_path, chunk_size=500, train=True)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    print(x.result())
        
