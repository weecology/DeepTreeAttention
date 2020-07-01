#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention import __file__ as ROOT
from DeepTreeAttention.generators import make_dataset
from DeepTreeAttention.utils.start_cluster import start
import os

att = AttentionModel()

#get root dir full path
client = start(cpus=50, mem_size="10GB")
train_tfrecords = make_dataset.generate_training(att.config["train"]["sensor_path"], att.config["train"]["ground_truth_path"], savedir=att.config["train"]["tfrecords"],use_dask=True,client=client, chunk_size=5000)
print("Created {} training records:{}...".format(len(train_tfrecords),train_tfrecords[0:3]))

#Currently just predict itself
predict_tfrecords = make_dataset.generate_prediction(att.config["train"]["sensor_path"], savedir="/orange/ewhite/b.weinstein/Houston2018/tfrecords/predict/",use_dask=True,client=client, chunk_size=5000)
print("Created {} prediction records:{}...".format(len(predict_tfrecords),predict_tfrecords[0:3]))
