#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention import __file__ as ROOT
from DeepTreeAttention.generators import make_dataset
import os

att = AttentionModel()

#get root dir full path
topdir = os.path.dirname(os.path.dirname(ROOT))
tfrecord_dir = os.path.join(topdir,"data/processed")
tfrecords = make_dataset.generate(att.config["train"]["sensor_path"], att.config["train"]["ground_truth_path"], savedir=tfrecord_dir)
print("Created {} records:{}...".format(len(tfrecords),tfrecords[0:3]))