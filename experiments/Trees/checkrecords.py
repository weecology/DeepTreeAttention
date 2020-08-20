import glob
import numpy as np

from DeepTreeAttention.generators import boxes
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
labels=[]
for image, label in dataset:
    counter+=image.shape[0]
    labels.append(label)

labels = np.vstack(labels)
print("There are {} train labels".format(len(np.unique(labels))))

created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
labels = []
for image, label in dataset:
    counter+=image.shape[0]
    labels.append(label)

labels = np.vstack(labels)
print("There are {} test labels".format(len(np.unique(labels))))
    
    

print(counter)