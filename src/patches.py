#Patches
import rasterio
from rasterio.features import geometry_mask
import numpy as np

def crop(bounds, sensor_path, savedir = None, basename = None,suffix=None):
    """Given a 4 pointed bounding box, crop sensor data"""
    left, bottom, right, top = bounds 
    src = rasterio.open(sensor_path)        
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)) 
    res = src.res[0]
    height = (top - bottom)/res
    width = (right - left)/res  
    if savedir:
        if suffix:
            filename = "{}/{}_{}.tif".format(savedir, basename, suffix)
        else:
            filename = "{}/{}.tif".format(savedir, basename)
            
        with rasterio.open(filename, "w", driver="GTiff",height=height, width=width, count = img.shape[0], dtype=img.dtype) as dst:
            dst.write(img)
    if savedir:
        return filename
    else:
        return img    
    

