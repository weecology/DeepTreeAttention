import glob
import pandas as pd
import numpy as np

from DeepTreeAttention.generators import boxes
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/train/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
labels=[]
sites =[]
for (image, site), label in dataset:
    counter+=image.shape[0]
    labels.append(label)
    sites.append(site)

sites = np.concatenate(sites)
labels = np.vstack(labels)
labels = np.argmax(labels,1)
print("There are {} train labels".format(len(np.unique(labels))))

created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*.tfrecord")
dataset = boxes.tf_dataset(created_records, batch_size=100)
counter=0
test_labels = []
test_site = []
for (image, site), label in dataset:
    test_site.append(site)
    counter+=image.shape[0]
    test_labels.append(label)

test_site = np.concatenate(test_site)
test_labels = np.vstack(test_labels)
test_labels = np.argmax(test_labels,1)
print("There are {} test labels".format(len(np.unique(test_labels))))

df=pd.DataFrame({"label":test_labels,"site":test_site})
df[df.label==17].site.unique()

traindf = pd.DataFrame({"label":labels,"site":sites})
traindf[traindf.label==17].site.unique()
