import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from DeepTreeAttention.generators import boxes

#metadata
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*.tfrecord")
dataset = boxes.tf_dataset(created_records, mode="metadata",batch_size=256)
counter=0
labels=[]
data =[]
for metadata, label in dataset:
    counter+=metadata.shape[0]
    print(counter)
    labels.append(label)
    data.append(metadata)



created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, mode="ensemble",batch_size=1)
counter=0
labels=[]
data =[]
for (HSI, RGB), label in dataset:
    counter+=RGB.shape[0]
    labels.append(label)
    data.append(RGB)

labels = np.vstack(labels)
labels = np.argmax(labels,1)
print("There are {} train labels".format(len(np.unique(labels))))

#Hypergator
from DeepTreeAttention.generators import boxes
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, mode="ensemble",batch_size=100)
counter=0
labels=[]
data =[]
for (HSI, RGB), label in dataset:
    counter+=image.shape[0]
    labels.append(label)
    data.append(RGB)

labels = np.vstack(labels)
labels = np.argmax(labels,1)
print("There are {} train labels".format(len(np.unique(labels))))

created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
test_labels = []
test_elevation = []
for (image, elevation), label in dataset:
    test_elevation.append(elevation)
    counter+=image.shape[0]
    test_labels.append(label)

test_elevation = np.concatenate(test_elevation)
test_labels = np.vstack(test_labels)
test_labels = np.argmax(test_labels,1)
print("There are {} test labels".format(len(np.unique(test_labels))))

df=pd.DataFrame({"label":test_labels,"elevation":test_elevation})
df[df.label==17].elevation.unique()

traindf = pd.DataFrame({"label":labels,"elevation":elevations})
traindf[traindf.label==17].elevation.unique()
