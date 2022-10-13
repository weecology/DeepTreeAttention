import pandas as pd
import os
from skimage import io
import numpy as np
import cv2

csv_file="data/raw/dead_train.csv"
df = pd.read_csv(csv_file)
df = df[df["label"] == "Alive"]
root_dir="/Users/benweinstein/Documents/NeonTreeEvaluation/evaluation/RGB"
savedir = "/Users/benweinstein/Dropbox/Weecology/DeadTrees/TreeDetectionZooniverse/annotations/unsorted"
df["id"] = range(df.shape[0])
df["id"] = df.apply(lambda x: "{}_{}".format(x["image_name"],x["id"]) , axis=1)
df["site"] = df.image_name.apply(lambda x: x.str.split("_")[0])
df = df.groupby("site").apply(lambda x: x.head(20))
for index, selected_row in df.iterrows(): 
    img_name = os.path.join(root_dir, selected_row["image_path"])
    image = io.imread(img_name)

    # select annotations
    xmin, xmax, ymin, ymax = selected_row[["xmin","xmax","ymin","ymax"]].values.astype(int)
    
    xmin = np.max([0,xmin-10])
    xmax = np.min([image.shape[1],xmax+10])
    ymin = np.max([0,ymin-10])
    ymax = np.min([image.shape[0],ymax+10])
    
    box = image[ymin:ymax, xmin:xmax]
    cv2.imwrite("{}/{}.png".format(savedir, selected_row["id"]), box[:,:,::-1])
    