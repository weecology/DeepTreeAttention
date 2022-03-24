#data utils
import argparse
import rasterio as rio
import json
import numpy as np
from torchvision import transforms
from sklearn import preprocessing
import torch
import yaml
import warnings

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
    img = np.asarray(image, dtype='float32')
    data = img.reshape(img.shape[0], np.prod(img.shape[1:]))
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)    
        data  = preprocessing.scale(data)
    img = data.reshape(img.shape)
    
    if not channel_is_first:
        img = np.rollaxis(img, 2,0)
        
    normalized = torch.from_numpy(img)
    
    return normalized

def load_image(img_path, image_size):
    """Load and preprocess an image for training/prediction"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rio.errors.NotGeoreferencedWarning)
        image = rio.open(img_path).read()       
    image = preprocess_image(image, channel_is_first=True)
    
    #resize image
    image = transforms.functional.resize(image, size=(image_size,image_size), interpolation=transforms.InterpolationMode.NEAREST)
    
    return image

def my_collate(batch):
    """Drop empty batches"""
    batch = list(filter (lambda x:x[1]["HSI"] is not None, batch))
    
    return torch.utils.data.dataloader.default_collate(batch)

def ensemble(results, year_individuals):
    """combine a set of results from main.evaluate crowns"""
    #Average among years
    temporal_prediction = []
    temporal_score = []
    for x in results.individual:
        year_mean = np.mean(np.vstack(year_individuals[x]), axis=0)
        temporal_prediction.append(np.argmax(year_mean))
        temporal_score.append(np.max(year_mean))    
    
    results["temporal_pred_label_top1"] = temporal_prediction
    results["temporal_top1_score"] = temporal_score
    
    return results