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
    src = rasterio.open(fname)
    T0 = src.transform  # upper-left pixel corner affine transform
    A = src.read()  # pixel values
    
    # All rows and columns
    cols, rows = np.meshgrid(np.arange(A.shape[2]), np.arange(A.shape[1]))
    
    # Function to convert pixel row/column index (from 0) to easting/northing at centre
    rc2en = lambda r, c: (c, r) * T0
    
    # All eastings and northings (there is probably a faster way to do this)
    eastings, northings = np.vectorize(rc2en, otypes=[np.float, np.float])(cols, rows)
    
    #get values at each position to be absolutely sure they are the same
    labels=[]
    for x,y, in zip(eastings.flatten(), northings.flatten()):
        for val in src.sample([(x, y)]):
            labels.append(val[0])
        
    results = pd.DataFrame({"label":labels,"easting":eastings.flatten(),"northing":northings.flatten()})
    
    return results

def select_training_crops(infile, coordinates, size=5):
    """Generate a square window crop centered on a geographic location
    Args:
        infile: path to sensor data
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
            py, px = dataset.index(x=x, y=y)
            
            #Add to index unsure of x,y order! #TODO https://rasterio.readthedocs.io/en/latest/api/rasterio.io.html#rasterio.io.BufferedDatasetWriter.index
            raster_rows.append(py)
            raster_cols.append(px)
                        
            # Build an NxN window
            window = rasterio.windows.Window(px - size//2, py - size//2, size, size)
    
            # Read the data in the window
            # clip is a nbands * size * size numpy array
            crop = dataset.read(window=window, boundless=True)    
            crop = np.rollaxis(crop, 0, 3)
                        
            crops.append(crop)
            
    return crops, raster_rows, raster_cols

def select_prediction_crops(infile, index_iterable, size=5):
    """Generate a square window crop centered on a geographic location
    Args:
        index_iterable: a zipped (x,y) tuple of indices to process
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
        for x, y in index_iterable:
            
            raster_rows.append(x)
            raster_cols.append(y)
                        
            # Build an NxN window
            window = rasterio.windows.Window(x - size//2, y - size//2, size, size)
    
            # Read the data in the window
            # clip is a nbands * size * size numpy array
            crop = dataset.read(window=window, boundless=True)    
            crop = np.rollaxis(crop, 0, 3)            
            crops.append(crop)
            
    return crops, raster_rows, raster_cols

def _record_wrapper_(sensor_path, size, classes, filename, train, coordinates=None, index_iterable=None, labels=None):
    """A wrapper for writing the correct type of tfrecord (train=True or False)"""
    
    if train:
        if labels is None:
            raise ValueError("Missing labels but training mode set to True")
        sensor_patches, x, y = select_training_crops(sensor_path, coordinates, size=size)        
        create_tfrecords.write_tfrecord(
            filename=filename,
            images=sensor_patches,
            labels=labels,
            classes=classes,
            train=train)
    else:
        sensor_patches, x, y = select_prediction_crops(infile=sensor_path, index_iterable=index_iterable, size=size)        
        create_tfrecords.write_tfrecord(
            filename=filename,
            images=sensor_patches,
            x=x,
            y=y,
            classes=classes,
            train=train)        
    
    return filename

def generate_training(sensor_path,
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
    print("There are {} label pixels in the labeled ground truth".format(results.shape[0]))

    #Create chunks to write
    results["chunk"] = np.arange(len(results)) // chunk_size
    basename = os.path.splitext(os.path.basename(sensor_path))[0]
    filenames = []    

    if use_dask:
        if client is None:
            raise ValueError("use_dask is {} but no client specified".format(use_dask))

        for g, df in results.groupby("chunk"):
            coordinates = zip(df.easting, df.northing)            
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)  
            #Submit to dask client
            fn = client.submit(_record_wrapper_,
                               labels=df.label.values,
                               sensor_path=sensor_path,
                               coordinates=coordinates,
                               size=size,
                               classes=classes,
                               filename=filename,
                               train=True)
            filenames.append(fn)
        wait(filenames)
        filenames = [x.result() for x in filenames]

    else:
        for g, df in results.groupby("chunk"):
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)     
            coordinates = zip(df.easting, df.northing)

            #Write record
            fn = _record_wrapper_(
                labels=df.label.values,
                sensor_path=sensor_path,
                coordinates=coordinates,
                size=size,
                classes=classes,
                filename=filename,
                train=True)
            filenames.append(fn)

    return filenames

def generate_prediction(sensor_path,
                      size=11,
                      chunk_size = 500,
                      classes=20,
                      savedir=".",
                      use_dask=False,
                      client=None):
    """Yield one instance of data with raster indices
    Args:
        chunk_size: number of images per tfrecord
        size: N x N image size
        savedir: directory to save tfrecords
        use_dask: optional dask client to parallelize computation
    Returns:
        filename: tfrecords path
    """
    with rasterio.open(sensor_path) as src:
        cols, rows = np.meshgrid(np.arange(src.shape[1]), np.arange(src.shape[0]))
        results = pd.DataFrame({"rows":np.ravel(rows),"cols":np.ravel(cols)})
    
    #turn ground truth into a dataframe of coords
    print("There are {} sensor pixels in the prediction data".format(results.shape[0]))
    
    #Create chunks to write
    results["chunk"] = np.arange(len(results)) // chunk_size
    basename = os.path.splitext(os.path.basename(sensor_path))[0]
    filenames = []    

    if use_dask:
        if client is None:
            raise ValueError("use_dask is {} but no client specified".format(use_dask))
        
        for g, df in results.groupby("chunk"):
            coordinates = zip(df.rows,df.cols)            
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)  
            #Submit to dask client
            fn = client.submit(_record_wrapper_,
                               sensor_path=sensor_path,
                               index_iterable=coordinates,
                               size=size,
                               classes=classes,
                               filename=filename,
                               train=False)
            filenames.append(fn)
        wait(filenames)
        filenames = [x.result() for x in filenames]
        
    else:
        for g, df in results.groupby("chunk"):
            filename = "{}/{}_{}.tfrecord".format(savedir,basename,g)     
            coordinates = zip(df.rows,df.cols)
            
            #Write record
            fn = _record_wrapper_(
                sensor_path=sensor_path,
                index_iterable=coordinates,
                size=size,
                classes=classes,
                filename=filename,
                train=False)
            filenames.append(fn)
    
    return filenames
    
def tf_dataset(tfrecords,
               batch_size=2,
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
    
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
        
    if shuffle:    
        dataset = dataset.shuffle(buffer_size=batch_size*5)
    if train:
        dataset = dataset.map(create_tfrecords._train_parse_, num_parallel_calls=200)
    else:
        dataset = dataset.map(create_tfrecords._predict_parse_, num_parallel_calls=200)
    #batch
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=AUTO)

    return dataset
