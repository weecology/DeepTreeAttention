#data utils
import argparse
import rasterio as rio
import json
import os
import numpy as np
from torchvision import transforms
from sklearn import preprocessing
import torch
import yaml
import warnings
import pandas as pd
from torch.utils.data.dataloader import default_collate
import os

def read_config(config_path):
    """Read config yaml file"""
    #Allow command line to override 
    parser = argparse.ArgumentParser("DeepTreeAttention config")
    parser.add_argument('-d', '--my-dict', type=json.loads, default=None)
    args = parser.parse_known_args()
    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    except Exception as e:
        raise FileNotFoundError("There is no config at {}, yields {}".format(
            config_path, e))
    
    #Update anything in argparse to have higher priority
    if args[0].my_dict:
        for key, value in args[0].my_dict:
            config[key] = value
        
    return config

def preprocess_image(image, channel_is_first=False):
    """Preprocess a loaded image, if already C*H*W set channel_is_first=True"""
    
    #Clip first and last 10 bands
    if image.shape[0] > 3:
        image = image[10:,:,:]
        image = image[:-10,:,:]
        
    img = np.asarray(image, dtype='float32')
    data = img.reshape(img.shape[0], np.prod(img.shape[1:])).T
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)    
        data  = preprocessing.minmax_scale(data, axis=1).T
    img = data.reshape(img.shape)
    
    if not channel_is_first:
        img = np.rollaxis(img, 2,0)
        
    normalized = torch.from_numpy(img)
    
    return normalized

def load_image(img_path, image_size):
    """Load and preprocess an image for training/prediction"""
    if os.path.splitext(img_path)[-1] == ".npy":
        try:
            image = np.load(img_path)
        except:
            raise ValueError("Cannot load {}".format(img_path))
        
    elif os.path.splitext(img_path)[-1] == ".tif":   
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', rio.errors.NotGeoreferencedWarning)
            image = rio.open(img_path).read()
    else:
        raise ValueError("image path must be .npy or .tif, found {}".format(img_path))
        
    image = preprocess_image(image, channel_is_first=True)
    
    #resize image
    image = transforms.functional.resize(image, size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST)
    
    return image

def my_collate(batch):
    batch = [x for x in batch if x[1]["HSI"] is not None]
    
    return default_collate(batch)

def predictions_to_df(predictions):
    """format a dataframe"""
    individuals = np.concatenate([x[0] for x in predictions])
    predictions = np.concatenate([x[1] for x in predictions])
    predictions = pd.DataFrame(predictions.squeeze())
    predictions["individual"] = individuals    
    
    return predictions
