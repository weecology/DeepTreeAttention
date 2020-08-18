import glob
from DeepTreeAttention.generators import boxes
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
for image, label in dataset:
    counter+=image.shape[0]
    