import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DeepTreeAttention.generators import boxes

#metadata
created_records = glob.glob("/orange/idtrees-collab/DeepTreeAttention/tfrecords/evaluation/*.tfrecord")
dataset = boxes.tf_dataset(created_records, mode = "metadata", batch_size=10)
counter=0
labels=[]
data =[]
for data, label in dataset:
    counter+=data.shape[0]
    print(counter)
