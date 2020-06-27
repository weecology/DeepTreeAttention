#### tf.data input pipeline ###
import numpy as np
import rasterio
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import pandas as pd

def _read_raster(path):
    """Read a hyperspetral raster .tif 
    Args:
        path: a path to a .tif hyperspectral raster
    Returns:
        src: a numpy array of height x width x channels
        """
    r = rasterio.open(path)
    src = r.read()

    return src

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
        for i, (lon, lat) in enumerate(coordinates):
    
            # Get pixel coordinates from map coordinates
            py, px = dataset.index(lon, lat)
            print('Pixel Y, X coords: {}, {}'.format(py, px))
    
            # Build an NxN window
            window = rasterio.windows.Window(px - size//2, py - size//2, size, size)
            print(window)
    
            # Read the data in the window
            # clip is a nbands * size * size numpy array
            crop = dataset.read(window=window)    
            crop = np.rollaxis(crop, 0, 3)
            
            #zero pad on border
            padded_crop = pad(crop, target_shape=(size,size,crop.shape[2]))
            
            crops.append(padded_crop)
            
    return crops
            
            
    
def tf_data_generator(sensor_path,
                      ground_truth_path,
                      size=11,
                      classes=20):
    """Yield one instance of data with one hot labels"""

    #turn ground truth into a dataframe of coords
    results = get_coordinates(ground_truth_path.decode())
    coordinates = zip(results.easting, results.northing)
    sensor_patches = select_crops(sensor_path.decode(), coordinates, size=size)
    
    #Turn data labels into one-hot
    label_onehot = to_categorical(results.label.values, num_classes=classes)
    zipped_data = zip(sensor_patches, label_onehot)

    while True:
        for data, label in zipped_data:
            yield data, label

def tf_dataset(sensor_path,
               ground_truth_path,
               size=11,
               batch_size=1,
               classes=20,
               repeat=True,
               shuffle=True):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        sensor_list: file path to sensor data .tif
        ground_truth_path: file path to ground truth data .tif
        repeat: Should the dataset repeat infinitely (e.g. training)
    Returns:
        dataset: a tf.data dataset yielding crops and labels
        """

    #Get data from generator
    dataset = tf.data.Dataset.from_generator(
        tf_data_generator,
        args=[sensor_path, ground_truth_path, size,classes],
        output_types=(tf.float32, tf.uint8),
        output_shapes=((size, size, None), (classes)))

    #batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()

    return dataset
