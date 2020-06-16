#### tf.data input pipeline ###
import tensorflow as tf
import rasterio

from DeepTreeAttention.generators.extract_patches import extract_patches

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


def tf_data_generator(sensor_path, ground_truth_path, crop_height=11,crop_width=11, sensor_channels=48):
    """Yield one instance of data with labels"""
    
    #read and extract patches
    sensor_array = _read_raster(sensor_path.decode())
    label_array = _read_raster(ground_truth_path.decode())
    
    sensor_patches = extract_patches(sensor_array, crop_width, crop_height)
    sensor_patches = tf.reshape(sensor_patches,[-1,crop_width,crop_height,sensor_channels])
    
    label_patches = extract_patches(label_array, 1, 1)
    label_patches = tf.reshape(label_patches,[-1,1])
    
    zipped_data = zip(sensor_patches, label_patches)

    while True:
        for data, label in zipped_data:
            yield data, label

def tf_dataset(sensor_path, ground_truth_path, crop_height=11,crop_width=11, sensor_channels=48, batch_size=1):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        sensor_list: file path to sensor data .tif
        ground_truth_path: file path to ground truth data .tif
    Returns:
        dataset: a tf.data dataset yielding crops and labels
        """
    
    #Get data from generator
    dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [sensor_path, ground_truth_path, crop_height, crop_width, sensor_channels],output_types = (tf.float32, tf.uint8),
                                             output_shapes = ((crop_width,crop_height,sensor_channels), (1)))

    #batch
    dataset = dataset.shuffle(buffer_size = 100)    
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.repeat()
    
    return dataset
    
    
    