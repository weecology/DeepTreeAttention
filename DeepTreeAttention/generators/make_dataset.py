#### tf.data input pipeline ###
import numpy as np
import os
import rasterio
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd
from .import create_tfrecords

def get_coordinates(fname):   
    """Read raster and convert into a zipped list of values and coordinates
    Args:
        fname: path to raster on file
    Returns:
        results: a zipped list that yields values, coordinates in a tuple
        """
    with rasterio.open(fname) as r:
        T0 = r.transform  # upper-left pixel corner affine transform
        A = r.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T0
    
    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(rows, cols)
    
    results = pd.DataFrame({"label":A.flatten(),"easting":eastings.flatten(),"northing":northings.flatten()})
    
    return results

def pad(array, target_shape):
    result = np.zeros(target_shape)
    result[:array.shape[0],:array.shape[1]] = array
    
    return result

def select_crops(infile, coordinates, size=5):
    """Generate a square window crop centered on a geographic location
    Args:
        infile: path to raster
        coordinates: a tuple of (easting, northing)
        size: number of pixels to buffer (expressed as a diameter, not a radius). 
    Returns:
        crop: a numpy array of cropped values of size (2*crop_height + 1, 2*crop_width+1)
    """
    # Open the raster
    crops = []
    with rasterio.open(infile) as dataset:
    
        # Loop through your list of coords
        for i, (x, y) in enumerate(coordinates):
    
            # Get pixel coordinates from map coordinates
            py, px = dataset.index(x, y)
    
            # Build an NxN window
            window = rasterio.windows.Window(px - size//2, py - size//2, size, size)
    
            # Read the data in the window
            # clip is a nbands * size * size numpy array
            crop = dataset.read(window=window)    
            crop = np.rollaxis(crop, 0, 3)
            
            #zero pad on border
            padded_crop = pad(crop, target_shape=(size,size,crop.shape[2]))
            
            crops.append(padded_crop)
            
    return crops

def generate(sensor_path,
                      ground_truth_path,
                      size=11,
                      chunk_size = 500,
                      classes=20,
                      savedir="."):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of images per tfrecord
        size: N x N image size
        savedir: directory to save tfrecords
    Returns:
        filename: tfrecords path
    """

    #turn ground truth into a dataframe of coords
    results = get_coordinates(ground_truth_path)
    print("There are {} label pixels".format(results.shape[0]))
    
    #Create groups of 100 to save to tfrecords
    results["chunk"] = np.arange(len(results)) // chunk_size
    counter = 0
    basename = os.path.splitext(os.path.basename(sensor_path))[0]
    filenames = []
    for g, df in results.groupby("chunk"):
        print(df.shape)
        coordinates = zip(df.easting, df.northing)    
        #Crop
        sensor_patches = select_crops(sensor_path, coordinates, size=size)
        print("Finished cropping {} sensor images".format(len(sensor_patches)))       
        
        filename = "{}/{}_{}.tfrecord".format(savedir,basename,counter)
        create_tfrecords.write_tfrecord(filename, sensor_patches, results.label.values)
        filenames.append(filename)
        counter +=1
    
    return filenames
    
def tf_dataset(tfrecords,
               batch_size=1,
               repeat=True,
               shuffle=True):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        tfrecords: path to tfrecords, see generate.py
        repeat: Should the dataset repeat infinitely (e.g. training)
    Returns:
        dataset: a tf.data dataset yielding crops and labels
        """

    dataset = tf.data.TFRecordDataset(filepath)

    #batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()

    return dataset
