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
from glob import glob
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

def create_glob_lists(config):
    """Creating glob lists is expensive, do it only once at the beginning of the run."""
    rgb_pool = glob(config["rgb_sensor_pool"], recursive=True)
    rgb_pool = [x for x in rgb_pool if "neon-aop-products" not in x]
    rgb_pool = [x for x in rgb_pool if "classified" not in x]    
    rgb_pool = [x for x in rgb_pool if not "point_cloud" in x]
    rgb_pool = [x for x in rgb_pool if not "UTM" in x]
    
    
    
    h5_pool = glob(config["HSI_sensor_pool"], recursive=True)
    h5_pool = [x for x in h5_pool if not "neon-aop-products" in x]
    h5_pool = [x for x in h5_pool if not "point_cloud" in x]
    h5_pool = [x for x in h5_pool if not "products" in x]
    
    hsi_pool = glob("{}/*.tif".format(config["HSI_tif_dir"]))
    try:
        CHM_pool = glob(config["CHM_pool"], recursive=True)
    except:
        CHM_pool = None
    
    return rgb_pool, h5_pool, hsi_pool, CHM_pool

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

def resize_or_pad(image, image_size, pad=False):
    """Resize an image to a square size, or pad with zeros to a square size. This is useful for creating batches with varying size data
    Args:
        image: a numpy array
        image_size: pixel width or height
        pad: Pad with zeros instead of resizing
    """
    
    if image_size is not -1:
        image = transforms.functional.resize(image, size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST)
    else:
        pad_height =  image.shape[1]
        pad_width = image.shape[2]
        image = transforms.functional.pad(image, padding=[pad_width, pad_height])
    return image

def load_image(img_path=None, image_size=30, pad=True):
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
    image = resize_or_pad(image, image_size, pad=pad)

    return image

def my_collate(batch):
    batch = [x for x in batch if x[1]["HSI"] is not None]
    
    return default_collate(batch)

def skip_none_collate(batch):
    batch = [x for x in batch if x is not None]
    return default_collate(batch)    
    
def predictions_to_df(predictions):
    """format a dataframe"""
    individuals = np.concatenate([x[0] for x in predictions])
    predictions = np.concatenate([x[1] for x in predictions])
    predictions = pd.DataFrame(predictions.squeeze())
    predictions["individual"] = individuals    
    
    return predictions

def preload_image_dict(df, config):
    """Load an entire dataset into memory and place it on device. This is useful for point to objects already in memory
    Args:
        df: a pandas dataframe with individual, tile_year and image_path columns for each image on disk
        config: a DeepTreeAttention config
    """
    years = df.tile_year.unique()    
    individuals = df.individual.unique()
    image_paths = df.groupby("individual").apply(lambda x: x.set_index('tile_year').image_path.to_dict())    
    image_dict = { }
    for individual in individuals:
        images = { }
        ind_annotations = image_paths[individual]
        for year in years:
            try:
                year_annotations = ind_annotations[year]
            except KeyError:
                images[str(year)] = image = torch.zeros(config["bands"], config["image_size"],  config["image_size"])  
                continue
            image_path = os.path.join(config["crop_dir"], year_annotations)
            image = load_image(image_path, image_size=config["image_size"])                        
            images[str(year)] = image
        image_dict[individual] = images 
    
    return image_dict

