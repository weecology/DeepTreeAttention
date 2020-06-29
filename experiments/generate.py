#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention import __file__ as ROOT
from DeepTreeAttention.generators import make_dataset
import os

att = AttentionModel()

#get root dir full path
tfrecords = make_dataset.generate(att.config["train"]["sensor_path"], att.config["train"]["ground_truth_path"], savedir=att.config["train"]["tfrecords"])
print("Created {} records:{}...".format(len(tfrecords),tfrecords[0:3]))