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
        raster_rows: numpy row index
        raster_cols: numpy col index
    """
    # Open the raster
    crops = []
    raster_rows = [ ]
    raster_cols = [ ]
    with rasterio.open(infile) as dataset:
    
        # Loop through your list of coords
        for i, (x, y) in enumerate(coordinates):
    
            # Get pixel coordinates from map coordinates
            py, px = dataset.index(x, y)
            
            #Add to index unsure of x,y order! #TODO https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.BufferedDatasetWriter.index
            raster_rows.append(py)
            raster_cols.append(px)
                        
            # Build an NxN window
            window = rasterio.windows.Window(px - size//2, py - size//2, size, size)
    
            # Read the data in the window
            # clip is a nbands * size * size numpy array
            crop = dataset.read(window=window)    
            crop = np.rollaxis(crop, 0, 3)
            
            #zero pad on border
            padded_crop = pad(crop, target_shape=(size,size,crop.shape[2]))
            
            crops.append(padded_crop)
            
    return crops, raster_rows, raster_cols

def _record_wrapper_(results, sensor_path, coordinates, size, classes, filename, train, x=None, y=None):
    """A wrapper for writing the correct type of tfrecord (train=True or False)"""
    sensor_patches, x, y = select_crops(sensor_path, coordinates, size=size)
    if train:
        create_tfrecords.write_tfrecord(
            filename=filename,
            images=sensor_patches,
            labels=results.label.values,
            classes=classes,
            train=train)
    else:
        create_tfrecords.write_tfrecord(
            filename=filename,
            images=sensor_patches,
            x=x,
            y=y,
            classes=classes,
            train=train)        
    
    return filename

def generate(sensor_path,
                      ground_truth_path,
                      size=11,
                      chunk_size = 500,
                      classes=20,
                      savedir=".",
                      use_dask=False,
                      client=None,
                      train=True):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of images per tfrecord
        size: N x N image size
        savedir: directory to save tfrecords
        use_dask: optional dask client to parallelize computation
        train: generate training records with labels (True) or prediction records with indices (False)
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
            #Submit to dask client
            fn = client.submit(_record_wrapper_,
                               results=df,
                               sensor_path=sensor_path,
                               coordinates=coordinates,
                               size=size,
                               classes=classes,
                               filename=filename,
                               train=train)
            filenames.append(fn)
        wait(filenames)
        filenames = [x.result() for x in filenames]
        
    else:
        for g, df in results.groupby("chunk"):
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)     
            coordinates = zip(df.easting,df.northing)
            
            #Write record
            fn = _record_wrapper_(
                results=df,
                sensor_path=sensor_path,
                coordinates=coordinates,
                size=size,
                classes=classes,
                filename=filename,
                train=train)
            filenames.append(fn)
    
    return filenames
    
def tf_dataset(tfrecords,
               batch_size=2,
               repeat=True,
               shuffle=True,
               train=True):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        tfrecords: path to tfrecords, see generate.py
        repeat: Should the dataset repeat infinitely (e.g. training)
        train: training mode -> records include training labels
    Returns:
        dataset: a tf.data dataset yielding crops and labels for train: True, crops and raster indices for train: False
        """

    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False
    
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=10)
    dataset = dataset.with_options(ignore_order)
    dataset = tf.data.TFRecordDataset(tfrecords)
        
    if shuffle:    
        dataset = dataset.shuffle(buffer_size=AUTO)
    if train:
        dataset = dataset.map(create_tfrecords._train_parse_)
    else:
        dataset = dataset.map(create_tfrecords._predict_parse_)
    #batch
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=AUTO)
    if repeat:
        dataset = dataset.repeat()

    return dataset
