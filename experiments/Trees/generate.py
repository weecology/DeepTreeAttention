#Generate tfrecords
from DeepTreeAttention.trees import AttentionModel
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.utils.start_cluster import start
from distributed import wait

import glob
import os

att = AttentionModel(config="/home/b.weinstein/DeepTreeAttention/conf/tree_config.yml")

#get root dir full path
client = start(cpus=10, mem_size="5GB") 

#Generate training data
shapefiles = glob.glob(os.path.join("/orange/idtrees-collab/DeepTreeAttention/WeakLabels/","*.shp"))
train_tfrecords = client.map(att.generate, shapefiles,chunk_size=10000, train=True)
print("Created {} training records:{}...".format(len(train_tfrecords),train_tfrecords[0:3]))

#Generate prediction data
#shapefiles = glob.glob(os.path.join("/orange/idtrees-collab/predictions/","*.shp"))
#predict_tfrecords = client.map(boxes.generate, shapefile=shapefiles,chunk_size=10000, train=True)
#print("Created {} prediction records:{}...".format(len(predict_tfrecords),predict_tfrecords[0:3]))

wait(train_tfrecords)
for x in train_tfrecords:
    try:
        train_tfrecords.result()
    except Exception as e:
        print(e)

#wait(predict_tfrecords)
