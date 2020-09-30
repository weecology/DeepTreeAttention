import glob
import pandas as pd
import numpy as np

from DeepTreeAttention.generators import boxes
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
labels=[]
elevations =[]
for (image, elevation), label in dataset:
    counter+=image.shape[0]
    labels.append(label)
    elevations.append(elevation)

elevations = np.concatenate(elevations)
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
