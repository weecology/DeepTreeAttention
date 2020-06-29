#Generate tfrecords
from DeepTreeAttention.main import AttentionModel
from DeepTreeAttention.generators import make_dataset

att = AttentionModel()

tfrecords = make_dataset.generate(att.config["train"]["sensor_path"], att.config["train"]["ground_truth_path"], savedir="../data/processed/")
print("Created {} records:{}...".format(len(tfrecords),tfrecords[0:3]))