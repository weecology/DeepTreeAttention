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
client = start(cpus=10, mem_size="5GB") 

#Generate training data
train_tfrecords = []
shapefiles = glob.glob(os.path.join("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/","*.shp"))
for shapefile in shapefiles:
    sensor_path = lookup_and_convert(shapefile, rgb_pool=att.config["train"]["rgb_sensor_pool"], hyperspectral_pool=att.config["train"]["hyperspectral_sensor_pool"], savedir=att.config["hyperspectral_tif_dir"])
    future = client.submit(att.generate, shapefile=shapefile, sensor_path=sensor_path, chunk_size=10000, train=True)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    x.result()
        
