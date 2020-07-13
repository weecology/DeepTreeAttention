#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.generators import make_dataset
from DeepTreeAttention.utils.start_cluster import start
import os

att = AttentionModel()

#get root dir full path
client = start(cpus=60, mem_size="5GB")
train_tfrecords = make_dataset.generate_training(sensor_path=att.config["train"]["sensor_path"], ground_truth_path=att.config["train"]["ground_truth_path"], savedir=att.config["train"]["tfrecords"],use_dask=True,client=client, chunk_size=20000)
print("Created {} training records:{}...".format(len(train_tfrecords),train_tfrecords[0:3]))

#Predict is the full tile raster
predict_tfrecords = make_dataset.generate_prediction(sensor_path="/home/b.weinstein/DeepTreeAttention/data/processed/20170218_UH_CASI_S4_NAD83.tif", savedir="/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/",use_dask=True,client=client, chunk_size=20000)
print("Created {} prediction records:{}...".format(len(predict_tfrecords),predict_tfrecords[0:3]))
