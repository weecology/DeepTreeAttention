#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention import __file__ as ROOT
from DeepTreeAttention.generators import make_dataset
import os

att = AttentionModel()

#get root dir full path
client = start(cpus=30, mem_size="10GB")
tfrecords = make_dataset.generate(att.config["train"]["sensor_path"], att.config["train"]["ground_truth_path"], savedir=att.config["train"]["tfrecords"],use_dask=True,client=client)
print("Created {} records:{}...".format(len(tfrecords),tfrecords[0:3]))