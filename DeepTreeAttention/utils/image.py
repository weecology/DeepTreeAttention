#Image utils
import cv2
import math
import rasterio
import numpy as np

def image_normalize(image):
    """normalize a 3d numpy array simiiar to tf.image.per_image_standardization"""
    mean = image.mean()
    stddev = image.std()
    adjusted_stddev = max(stddev, 1.0/math.sqrt(image.size))
    standardized_image = (image - mean) / adjusted_stddev
    
    return standardized_image

def resize(img, height, width):
    # resize image
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_NEAREST)

    return resized

def crop_image(src, box, expand=0): 
    """Read sensor data and crop a bounding box
    Args:
        src: a rasterio opened path
        box: geopandas geometry polygon object
        expand: add padding in percent to the edge of the crop
    Returns:
        masked_image: a crop of sensor data at specified bounds
    """
    #Read data and mask
    try:    
        left, bottom, right, top = box.bounds
        
        expand_width = (right - left) * expand /2
        expand_height = (top - bottom) * expand / 2
        
        #If expand is greater than increase both size
        if expand >= 0:
            expanded_left = left - expand_width
            expanded_bottom = bottom - expand_height
            expanded_right = right + expand_width
            expanded_top =  top+expand_height
        else:
            #Make sure of no negative boxes
            expanded_left = left+expand_width
            expanded_bottom = bottom+expand
            expanded_right = right-expand_width
            expanded_top =  top-expand_height            
        
        window = rasterio.windows.from_bounds(expanded_left, expanded_bottom, expanded_right, expanded_top, transform=src.transform)
        masked_image = src.read(window=window)
    except Exception as e:
        raise ValueError("sensor path: {} failed at reading window {} with error {}".format(src, box.bounds,e))
        
    #Roll depth to channel last
    masked_image = np.rollaxis(masked_image, 0, 3)
    
    #Skip empty frames
    if masked_image.size ==0:
        raise ValueError("Empty frame crop for box {} in sensor path {}".format(box, src))
        
    return masked_image