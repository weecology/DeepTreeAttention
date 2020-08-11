#### tf.data input pipeline ###
import geopandas
import numpy as np
import os
import pandas as pd
import rasterio
import tensorflow as tf
import cv2

from DeepTreeAttention.generators import create_tfrecords
from rasterio.windows import from_bounds

def resize(img, height, width):
    # resize image
    dim = (width, height)   
    img = img.astype("float32")
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    
    return resized
    
def generate_tfrecords(shapefile, sensor_path,
                      chunk_size=1000,
                      savedir=".",
                      height=40,
                      width=40,
                      classes=20,
                      train=True):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of windows per tfrecord
        savedir: directory to save tfrecords
        train: training mode to include yielded labels
    Returns:
        filename: tfrecords path
    """
    gdf = geopandas.read_file(shapefile)
    basename = os.path.splitext(os.path.basename(shapefile))[0]
    src = rasterio.open(sensor_path)
    
    gdf["box_index"] = ["{}_{}".format(basename,x) for x in gdf.index.values]
    labels = []
    crops = []
    indices = []
    for index, row in gdf.iterrows():
        #Add training label, ignore unclassified 0 class
        if train:
            if row["label"] == 0:
                continue
            
            labels.append(row["label"])
            
        window=from_bounds(*row["geometry"].bounds, transform = src.transform)
        masked_image = src.read(window=window)
        
        #Roll depth to channel last
        masked_image = np.rollaxis(masked_image, 0, 3)
        
        #Skip empty frames
        if masked_image.size ==0:
            continue
        
        crops.append(masked_image)
        indices.append(row["box_index"])
        
    #get keys and divide into chunks for a single tfrecord
    filenames = []
    counter = 0
    for i in range(0, len(crops)+1, chunk_size):
        chunk_crops = crops[i:i + chunk_size]
        chunk_index = indices[i:i + chunk_size]
        
        if train:
            chunk_labels = labels[i:i + chunk_size]
        else:
            chunk_labels = None
        
        #resize crops
        resized_crops = [resize(x, height, width) for x in chunk_crops]
        
        filename = "{}/{}_{}.tfrecord".format(savedir, basename, counter)
        write_tfrecord(filename=filename,
                                            images=resized_crops,
                                            labels=chunk_labels,
                                            indices=chunk_index,
                                            classes=classes)
        
        filenames.append(filename)
        counter +=1

    return filenames

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(filename, images, indices, labels=None, classes=21):
    """Write a training or prediction tfrecord
        Args:
            train: True -> create a training record with labels. False -> a prediciton record with raster indices
        """
    writer = tf.io.TFRecordWriter(filename)
    
    if labels is not None:
        #Write parser
        for index, image in enumerate(images):
            tf_example = create_record(index=indices[index], image=images[index], label=labels[index], classes=classes)
            writer.write(tf_example.SerializeToString())
    else:
        for index, image in enumerate(images):
            tf_example = create_record(index=indices[index], image=image, classes=classes)
            writer.write(tf_example.SerializeToString())

    writer.close()

def create_record(image, index, classes, label=None):
    """
    Generate one record from an image 
    Args:
        image: a numpy arry in the form height, row, depth channels
        index: box_index GIS label
        classes: number of classes of labels to train/predict
        label: Optional label for training class
    Returns:
        tf example parser
    """
    rows = image.shape[0]
    cols = image.shape[1]
    depth = image.shape[2]
    
    if label is not None:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'box_index': _bytes_feature(index.encode()),
                'image/data': _bytes_feature(image.tostring()),
                'label': _int64_feature(label),
                'image/height': _int64_feature(rows),
                'image/width': _int64_feature(cols),
                'image/depth': _int64_feature(depth),
                'classes': _int64_feature(classes),
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'box_index': _bytes_feature(index.encode()),
                'image/data': _bytes_feature(image.tostring()),
                'image/height': _int64_feature(rows),
                'image/width': _int64_feature(cols),
                'image/depth': _int64_feature(depth),
                'classes': _int64_feature(classes),
            }))        

    # Serialize to string and write to file
    return example

def _train_parse_(tfrecord):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        'box_index': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    classes = tf.cast(example['classes'], tf.int32)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)

    #recast
    label = tf.cast(example['label'], tf.int64)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.float32)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")

    #one hot
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_image, one_hot_labels

def _train_submodel_parse_(tfrecord):
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        'box_index': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    classes = tf.cast(example['classes'], tf.int32)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)

    #recast
    label = tf.cast(example['label'], tf.int64)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.float32)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")

    #one hot
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_image, (one_hot_labels, one_hot_labels, one_hot_labels)

def _predict_parse_(tfrecord):
    """Tfrecord parser for prediction. No labels available
        Args:
            tfrecord: path to tfrecord
        Returns:
            indices: x,y index of row, col position in original raster
            loaded_image: image data crop
        """
    # Define features
    features = {
        'image/data': tf.io.FixedLenFeature([], tf.string),
        'box_index': tf.io.FixedLenFeature([], tf.string),        
        "image/height": tf.io.FixedLenFeature([], tf.int64),
        "image/width": tf.io.FixedLenFeature([], tf.int64),
        "image/depth": tf.io.FixedLenFeature([], tf.int64),
        "classes": tf.io.FixedLenFeature([], tf.int64),
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)

    height = tf.cast(example['image/height'], tf.int64)
    width = tf.cast(example['image/width'], tf.int64)
    depth = tf.cast(example['image/depth'], tf.int64)

    # Load image from file
    image = tf.io.decode_raw(example['image/data'], tf.float32)
    image_shape = tf.stack([height, width, depth])

    # Reshape to known shape
    loaded_image = tf.reshape(image, image_shape, name="cast_loaded_image")

    return loaded_image, example['box_index']

def tf_dataset(tfrecords, batch_size=2, height=20, width=20, shuffle=True, mode="train", cores=10):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        tfrecords: path to tfrecords, see generate.py
        mode:  "train" mode records include training labels, "submodel" triples the layers to match number of softmax layers,  "predict" is just image data and coordinates
        height: crop resized height
        width: crop resized width
    Returns:
        dataset: a tf.data dataset yielding crops and labels for train: True, crops and raster indices for train: False
        """
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=cores)
    dataset = dataset.with_options(ignore_order)

    if shuffle:
        print("Shuffling data")
        dataset = dataset.shuffle(buffer_size=20)        
    
    if mode == "train":
        dataset = dataset.map(_train_parse_, num_parallel_calls=cores)
        dataset = dataset.shuffle(buffer_size=100)                
        dataset = dataset.batch(batch_size=batch_size)
        
    elif mode=="predict":
        dataset = dataset.map(_predict_parse_, num_parallel_calls=cores)
        dataset = dataset.batch(batch_size=batch_size)
        
    elif mode=="submodel":
        dataset = dataset.map(create_tfrecords._train_submodel_parse_, num_parallel_calls=cores)
        dataset = dataset.shuffle(buffer_size=100)                
        dataset = dataset.batch(batch_size=batch_size)
    else:
        raise ValueError("invalid mode, please use train, predict or submodel: {}".format(mode))
    
    dataset = dataset.prefetch(buffer_size=1)

    return dataset