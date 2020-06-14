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
    sensor_array = _read_raster(sensor_path)
    label_array = _read_raster(ground_truth_path)
    
    sensor_patches = extract_patches(sensor_array, crop_width, crop_height)
    sensor_patches = tf.reshape(sensor_patches,[-1,crop_width,crop_height,sensor_channels])
    
    label_patches = extract_patches(label_array, 1, 1)
    label_patches = tf.reshape(label_patches,[-1,1])
    
    zipped_data = zip(sensor_patches, label_patches)

    while True:
        for data, label in zipped_data:
            yield data, label

def training_dataset(sensor_list, ground_truth_list, crop_height=11,crop_width=11, channels=48, batch_size=1):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        sensor_list: a list of file paths to sensor data .tif
        ground_truth_list: a list of file paths to ground truth data .tif
    Returns:
        dataset: a tf.data dataset yielding crops and labels
        """
    
    #Get sensor data patches
    dataset = tf.data.Dataset.from_generator(tf_data_generator,args= [files, labels],output_types = (tf.float32, tf.uint8),
                                             output_shapes = ((None,crop_width,crop_height,channels), (None)))
    
    sensor_dataset = sensor_dataset.map(_extract_crop_, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    #Get label ground truth 
    label_dataset = tf.data.Dataset.from_tensor_slices(ground_truth_list)
    label_dataset = label_dataset.map(_read_raster, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    label_dataset = label_dataset.map(_extract_center_, num_parallel_calls=tf.data.experimental.AUTOTUNE)    
    
    #zip?
    combined_dataset  = sensor_dataset.zip(label_dataset)
    combined_dataset = combined_dataset.batch(batch_size=batch_size)
    
    #unbatch?
    #dataset = dataset.shuffle(buffer_size = 100)
    #dataset = dataset.repeat()
    #dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    
    return combined_dataset
    
    
    