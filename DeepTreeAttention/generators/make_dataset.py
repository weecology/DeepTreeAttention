#### tf.data input pipeline ###
from dask.distributed import wait        
import numpy as np
import os
import pandas as pd
import rasterio
import tensorflow as tf

from .import create_tfrecords
from DeepTreeAttention.utils.start_cluster import start

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

def _record_wrapper_(results, sensor_path, coordinates, size, classes, filename):
    sensor_patches = select_crops(sensor_path, coordinates, size=size)
    create_tfrecords.write_tfrecord(filename, sensor_patches, results.label.values, classes)
    
    return filename

def generate(sensor_path,
                      ground_truth_path,
                      size=11,
                      chunk_size = 500,
                      classes=20,
                      savedir=".",
                      use_dask=False,
                      client=None):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of images per tfrecord
        size: N x N image size
        savedir: directory to save tfrecords
        use_dask: optional dask client to parallelize computation
    Returns:
        filename: tfrecords path
    """
    #turn ground truth into a dataframe of coords
    results = get_coordinates(ground_truth_path)
    print("There are {} label pixels".format(results.shape[0]))
    
    #Create chunks to write
    results["chunk"] = np.arange(len(results)) // chunk_size
    basename = os.path.splitext(os.path.basename(sensor_path))[0]
    filenames = []    
    if use_dask:
        if client is None:
            raise ValueError("use_dask is {} but no client specified".format(use_dask))
        for g, df in results.groupby("chunk"):
            coordinates = zip(df.easting,df.northing)            
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)  
            fn = client.submit(_record_wrapper_, df, sensor_path, coordinates, size, classes, filename)
            filenames.append(fn)
        wait(filenames)
        filenames = [x.result() for x in filenames]
    else:
        for g, df in results.groupby("chunk"):
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)     
            coordinates = zip(df.easting,df.northing)
            fn = _record_wrapper_(df, sensor_path, coordinates, size, classes, filename)
            filenames.append(fn)
    
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

    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=5)
    dataset = dataset.with_options(ignore_order)
    
    if shuffle:    
        dataset = dataset.shuffle(buffer_size=AUTO)
    dataset = dataset.map(create_tfrecords._parse_fn,num_parallel_calls=AUTO)
    #batch
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=AUTO)
    if repeat:
        dataset = dataset.repeat()

    return dataset
