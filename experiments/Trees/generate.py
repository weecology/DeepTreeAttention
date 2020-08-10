#Generate tfrecords
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils.start_cluster import start
from distributed import wait
from prepare_field_data import lookup_and_convert
import glob
import os

att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")

#get root dir full path
client = start(cpus=10, mem_size="5GB") 

#Generate training data
train_tfrecords = []
shapefiles = glob.glob(os.path.join("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/","*.shp"))
for shapefile in shapefiles:
    sensor_path = lookup_and_convert(shapefile, rgb_pool=self.config["train"]["rgb_sensor_pool"], hyperspectral_pool=self.config["train"]["hyperspectral_sensor_pool"])
    future = client.submit(att.generate, shapefile=shapefile, sensor_path=sensor_path, chunk_size=10000, train=True)
    train_tfrecords.append(future)
    
wait(train_tfrecords)
for x in train_tfrecords:
    try:
        x.result()
    except Exception as e:
        print(e)
