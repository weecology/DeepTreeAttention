#### tf.data input pipeline ###
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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


def tf_data_generator(sensor_path,
                      ground_truth_path,
                      crop_height=11,
                      crop_width=11,
                      sensor_channels=48,
                      classes=20):
    """Yield one instance of data with one hot labels"""

    #read and extract patches
    sensor_array = _read_raster(sensor_path.decode())
    label_array = _read_raster(ground_truth_path.decode())

    sensor_patches = extract_patches(sensor_array, crop_width, crop_height)
    print("patches extracted, reshaping")
    sensor_patches = tf.reshape(sensor_patches,
                                [-1, crop_width, crop_height, sensor_channels])

    label_patches = extract_patches(label_array, 1, 1)
    label_patches = tf.reshape(label_patches, [-1, 1])

    #Turn data labels into one-hot
    label_onehot = to_categorical(label_patches, num_classes=classes)
    zipped_data = zip(sensor_patches, label_onehot)

    while True:
        for data, label in zipped_data:
            yield data, label

def tf_dataset(sensor_path,
               ground_truth_path,
               crop_height=11,
               crop_width=11,
               sensor_channels=48,
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
        args=[sensor_path, ground_truth_path, crop_height, crop_width, sensor_channels],
        output_types=(tf.float32, tf.uint8),
        output_shapes=((crop_width, crop_height, sensor_channels), (classes)))

    #batch
    if shuffle:
        dataset = dataset.shuffle(buffer_size=10)
    dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    if repeat:
        dataset = dataset.repeat()

    return dataset
