#### tf.data input pipeline ###
import geopandas
import numpy as np
import os
import pandas as pd
import rasterio
import random
import tensorflow as tf
import cv2
from rasterio.windows import from_bounds

def resize(img, height, width):
    # resize image
    dim = (width, height)
    img = img.astype("float32")
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
            expanded_left = left-expand_width
            expanded_bottom = bottom-expand_height
            expanded_right = right+expand_width
            expanded_top =  top+expand_height
        else:
            #Make sure of no negative boxes
            expanded_left = left+expand_width
            expanded_bottom = bottom+expand
            expanded_right = right-expand_width
            expanded_top =  top-expand_height            
        
        window=rasterio.windows.from_bounds(expanded_left, expanded_bottom, expanded_right, expanded_top, transform=src.transform)
        masked_image = src.read(window=window)
    except Exception as e:
        raise ValueError("sensor path: {} failed at reading window {} with error {}".format(sensor_path, box.bounds,e))
        
    #Roll depth to channel last
    masked_image = np.rollaxis(masked_image, 0, 3)
    
    #Skip empty frames
    if masked_image.size ==0:
        raise ValueError("Empty frame crop for box {} in sensor path {}".format(box, sensor_path))
    
    return masked_image
    
def generate_tfrecords(shapefile,
                       HSI_sensor_path,
                       RGB_sensor_path,
                       site,
                       elevation,
                       heights,
                       species_label_dict,
                       chunk_size=1000,
                       savedir=".",
                       HSI_size=40,
                       RGB_size=40,
                       classes=20,
                       number_of_sites=23,
                       train=True,
                       extend_box=0,
                       shuffle=True):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of windows per tfrecord
        savedir: directory to save tfrecords
        site: metadata site label as integer
        elevation: height above sea level in meters
        heights: height in m
        label_dict: taxonID -> numeric label
        RGB_size: size in pixels of one side of image
        HSI_size: size in pixels of one side of image
        train: training mode to include yielded labels
        number_of_sites: total number of sites used for one-hot encoding
        extend_box: units in meters to expand DeepForest bounding box to give crop more context
    Returns:
        filename: tfrecords path
    """
    gdf = geopandas.read_file(shapefile)
    basename = os.path.splitext(os.path.basename(shapefile))[0]
    HSI_src = rasterio.open(HSI_sensor_path)
    RGB_src = rasterio.open(HSI_sensor_path)

    gdf["box_index"] = ["{}_{}".format(basename, x) for x in gdf.index.values]
    labels = []
    HSI_crops = []
    RGB_crops = []
    indices = []
    
    for index, row in gdf.iterrows():
        #Add training label, ignore unclassified 0 class
        if train:
            labels.append(row["label"])
        try:
            HSI_crop = crop_image(HSI_src, row["geometry"], extend_box)
            RGB_crop = crop_image(RGB_src, row["geometry"], extend_box)
        except Exception as e:
            print("row {} failed with {}".format(index, e))
            continue
        
        HSI_crops.append(HSI_crop)
        RGB_crops.append(RGB_crop)
        indices.append(row["box_index"])

    #If passes a species label dict
    if species_label_dict is None:
        #Create and save a new species and site label dict
        unique_species_labels = np.unique(labels)
        species_label_dict = {}
        for index, label in enumerate(unique_species_labels):
            species_label_dict[label] = index
        pd.DataFrame(species_label_dict.items(), columns=["taxonID","label"]).to_csv("{}/species_class_labels.csv".format(savedir))
    
    numeric_species_labels = [species_label_dict[x] for x in labels]

    #shuffle before writing to help with validation data split
    if shuffle:
        if train:
            z = list(zip(HSI_crops, RGB_crops, heights, indices, numeric_species_labels))
            random.shuffle(z)
            HSI_crops, RGB_crops, heights, indices, numeric_species_labels = zip(*z)

    #get keys and divide into chunks for a single tfrecord
    filenames = []
    counter = 0
    for i in range(0, len(HSI_crops) + 1, chunk_size):
        chunk_HSI_crops = HSI_crops[i:i + chunk_size]
        chunk_RGB_crops = RGB_crops[i:i + chunk_size]        
        chunk_index = indices[i:i + chunk_size]
        chunk_height = heights[i:i + chunk_size]
        
        #All records in a single shapefile are the same site
        chunk_sites = np.repeat(site, len(chunk_index))
        chunk_elevations = np.repeat(elevation, len(chunk_index))
        
        if train:
            chunk_labels = numeric_species_labels[i:i + chunk_size]
        else:
            chunk_labels = None

        #resize crops
        resized_HSI_crops = [resize(x, HSI_size, HSI_size).astype("int16") for x in chunk_HSI_crops]
        resized_RGB_crops = [resize(x, RGB_size, RGB_size).astype("int16") for x in chunk_RGB_crops]

        filename = "{}/{}_{}.tfrecord".format(savedir, basename, counter)
        
        write_tfrecord(filename=filename,
                       HSI_images=resized_HSI_crops,
                       RGB_images=resized_RGB_crops,
                       labels=chunk_labels,
                       sites=chunk_sites,
                       heights=chunk_height,
                       elevations= chunk_elevations,
                       indices=chunk_index,
                       number_of_sites=number_of_sites,
                       classes=classes)

        filenames.append(filename)
        counter += 1

    return filenames


def _float32_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write_tfrecord(filename, HSI_images, RGB_images, sites, elevations, heights, indices, labels=None, classes=21, number_of_sites=23):
    """Write a training or prediction tfrecord
        Args:
            train: True -> create a training record with labels. False -> a prediciton record with raster indices
        """
    writer = tf.io.TFRecordWriter(filename)

    if labels is not None:
        #Write parser
        for index, image in enumerate(HSI_images):
            tf_example = create_record(
                index=indices[index],
                site = sites[index],
                HSI_image = HSI_images[index],
                RGB_image = RGB_images[index],
                label=labels[index],
                height=heights[index],                
                elevation=elevations[index],
                number_of_sites=number_of_sites,
                classes=classes)
            writer.write(tf_example.SerializeToString())
    else:
        for index, image in enumerate(HSI_images):
            tf_example = create_record(
                index=indices[index],
                site = sites[index],
                elevation = elevations[index],
                HSI_image=image,
                height=heights[index],
                RGB_image = RGB_images[index],
                number_of_sites=number_of_sites,
                classes=classes)
            writer.write(tf_example.SerializeToString())

    writer.close()


def create_record(HSI_image, RGB_image, index, site, elevation, height, classes, number_of_sites, label=None):
    """
    Generate one record from an image 
    Args:
        HSI_image: a numpy arry in the form height, row, depth channels
        RGB_image: a numpy arry in the form height, row, depth channels
        index: box_index GIS label
        classes: number of classes of labels to train/predict
        sites: number of geographic sites in train/test to one-hot labels
        elevation: height above sea level in meters
        label: Optional label for training class
    Returns:
        tf example parser
    """
    HSI_rows = HSI_image.shape[0]
    HSI_cols = HSI_image.shape[1]
    HSI_depth = HSI_image.shape[2]
    
    RGB_rows = RGB_image.shape[0]
    RGB_cols = RGB_image.shape[1]
    RGB_depth = RGB_image.shape[2]

    if label is not None:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'box_index': _bytes_feature(index.encode()),
                'HSI_image/data': _bytes_feature(HSI_image.tostring()),
                'label': _int64_feature(label),
                'site': _int64_feature(site),    
                'elevation': _int64_feature(elevation),                                
                'HSI_image/height': _int64_feature(HSI_rows),
                'HSI_image/width': _int64_feature(HSI_cols),
                'HSI_image/depth': _int64_feature(HSI_depth),
                'RGB_image/data': _bytes_feature(RGB_image.tostring()),                                
                'RGB_image/height': _int64_feature(RGB_rows),
                'RGB_image/width': _int64_feature(RGB_cols),
                'RGB_image/depth': _int64_feature(RGB_depth),                
                'classes': _int64_feature(classes),                
                'number_of_sites': _int64_feature(number_of_sites),
                'height': _float32_feature(height)
            }))
    else:
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'box_index': _bytes_feature(index.encode()),
                'HSI_image/data': _bytes_feature(HSI_image.tostring()),
                'HSI_image/height': _int64_feature(HSI_rows),
                'HSI_image/width': _int64_feature(HSI_cols),
                'HSI_image/depth': _int64_feature(HSI_depth),
                'RGB_image/data': _bytes_feature(RGB_image.tostring()),                
                'RGB_image/height': _int64_feature(RGB_rows),
                'RGB_image/width': _int64_feature(RGB_cols),
                'RGB_image/depth': _int64_feature(RGB_depth),        
                'classes': _int64_feature(classes),
                'site': _int64_feature(site),               
                'elevation': _int64_feature(elevation),                                                
                'number_of_sites': _int64_feature(number_of_sites),
                'height': _float32_feature(height)
            }))

    # Serialize to string and write to file
    return example


def _train_parse_(tfrecord):
    # Define features
    features = {
        'HSI_image/data': tf.io.FixedLenFeature([], tf.string),
        'RGB_image/data': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "HSI_image/height": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/width": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/depth": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/height": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/width": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
        "height": tf.io.FixedLenFeature([], tf.float32),        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_height = tf.cast(example['HSI_image/height'], tf.int64)
    HSI_width = tf.cast(example['HSI_image/width'], tf.int64)
    HSI_depth = tf.cast(example['HSI_image/depth'], tf.int64)
    HSI_image = tf.io.decode_raw(example['HSI_image/data'], tf.uint16)
    HSI_image_shape = tf.stack([HSI_height,HSI_width, HSI_depth])
    
    # Reshape to known shape
    loaded_HSI_image = tf.reshape(HSI_image, HSI_image_shape, name="cast_loaded_HSI_image")
    loaded_HSI_image = tf.cast(loaded_HSI_image, dtype=tf.float32)
    
    # Load RGB image from file
    RGB_height = tf.cast(example['RGB_image/height'], tf.int64)
    RGB_width = tf.cast(example['RGB_image/width'], tf.int64)
    RGB_depth = tf.cast(example['RGB_image/depth'], tf.int64)
    RGB_image = tf.io.decode_raw(example['RGB_image/data'], tf.uint16)
    RGB_image_shape = tf.stack([RGB_height,RGB_width, RGB_depth])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(RGB_image, RGB_image_shape, name="cast_loaded_RGB_image")
    loaded_RGB_image = tf.cast(loaded_RGB_image, dtype=tf.float32)
    
    #Metadata and labels
    classes = tf.cast(example['classes'], tf.int32)
    
    #recast and scale to km    
    number_of_sites = tf.cast(example['number_of_sites'], tf.int32)    
    site = tf.cast(example['site'], tf.int64)    
    elevation = tf.cast(example['elevation'], tf.float32)
    elevation = elevation / 1000
    one_hot_sites = tf.one_hot(site, number_of_sites)
    
    #tree height
    height = tf.cast(example['height'], tf.float32)
    height = height / 100
    
    #one hot encoding
    label = tf.cast(example['label'], tf.int64)    
    one_hot_labels = tf.one_hot(label, classes)

    return (loaded_HSI_image, loaded_RGB_image, elevation, height, one_hot_sites), one_hot_labels

def _RGB_train_parse_(tfrecord):
    # Define features
    features = {
        'RGB_image/data': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "RGB_image/height": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/width": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load RGB image from file
    RGB_height = tf.cast(example['RGB_image/height'], tf.int64)
    RGB_width = tf.cast(example['RGB_image/width'], tf.int64)
    RGB_depth = tf.cast(example['RGB_image/depth'], tf.int64)
    RGB_image = tf.io.decode_raw(example['RGB_image/data'], tf.uint16)
    RGB_image_shape = tf.stack([RGB_height,RGB_width, RGB_depth])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(RGB_image, RGB_image_shape, name="cast_loaded_RGB_image")
    loaded_RGB_image = tf.cast(loaded_RGB_image, dtype=tf.float32)
    
    #Metadata and labels
    classes = tf.cast(example['classes'], tf.int32)
    
    #one hot encoding
    label = tf.cast(example['label'], tf.int64)    
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_RGB_image, one_hot_labels

def _HSI_train_parse_(tfrecord):
    # Define features
    features = {
        'HSI_image/data': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "HSI_image/height": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/width": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load HSI image from file
    HSI_height = tf.cast(example['HSI_image/height'], tf.int64)
    HSI_width = tf.cast(example['HSI_image/width'], tf.int64)
    HSI_depth = tf.cast(example['HSI_image/depth'], tf.int64)
    HSI_image = tf.io.decode_raw(example['HSI_image/data'], tf.uint16)
    HSI_image_shape = tf.stack([HSI_height,HSI_width, HSI_depth])
    
    # Reshape to known shape
    loaded_HSI_image = tf.reshape(HSI_image, HSI_image_shape, name="cast_loaded_HSI_image")
    loaded_HSI_image = tf.cast(loaded_HSI_image, dtype=tf.float32)
    
    #Metadata and labels
    classes = tf.cast(example['classes'], tf.int32)
    
    #one hot encoding
    label = tf.cast(example['label'], tf.int64)    
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_HSI_image, one_hot_labels

def _train_HSI_submodel_parse_(tfrecord):
    # Define features
    features = {
        'HSI_image/data': tf.io.FixedLenFeature([], tf.string),
        'RGB_image/data': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "HSI_image/height": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/width": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/depth": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/height": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/width": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_height = tf.cast(example['HSI_image/height'], tf.int64)
    HSI_width = tf.cast(example['HSI_image/width'], tf.int64)
    HSI_depth = tf.cast(example['HSI_image/depth'], tf.int64)
    HSI_image = tf.io.decode_raw(example['HSI_image/data'], tf.uint16)
    HSI_image_shape = tf.stack([HSI_height, HSI_width, HSI_depth])
    
    # Reshape to known shape
    loaded_HSI_image = tf.reshape(HSI_image, HSI_image_shape, name="cast_loaded_HSI_image")
    loaded_HSI_image = tf.cast(loaded_HSI_image, dtype=tf.float32)
    
    #Metadata and labels
    classes = tf.cast(example['classes'], tf.int32)
    
    #one hot encoding
    label = tf.cast(example['label'], tf.int64)    
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_HSI_image, (one_hot_labels,one_hot_labels,one_hot_labels)

def _train_RGB_submodel_parse_(tfrecord):
    # Define features
    features = {
        'RGB_image/data': tf.io.FixedLenFeature([], tf.string),        
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "RGB_image/height": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/width": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load RGB image from file
    RGB_height = tf.cast(example['RGB_image/height'], tf.int64)
    RGB_width = tf.cast(example['RGB_image/width'], tf.int64)
    RGB_depth = tf.cast(example['RGB_image/depth'], tf.int64)
    RGB_image = tf.io.decode_raw(example['RGB_image/data'], tf.uint16)
    RGB_image_shape = tf.stack([RGB_height,RGB_width, RGB_depth])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(RGB_image, RGB_image_shape, name="cast_loaded_RGB_image")
    loaded_RGB_image = tf.cast(loaded_RGB_image, dtype=tf.float32)
    
    #Metadata and labels
    classes = tf.cast(example['classes'], tf.int32)
    
    #one hot encoding
    label = tf.cast(example['label'], tf.int64)    
    one_hot_labels = tf.one_hot(label, classes)

    return loaded_RGB_image, (one_hot_labels,one_hot_labels,one_hot_labels)

def _predict_parse_(tfrecord):
    # Define features
    features = {
        'HSI_image/data': tf.io.FixedLenFeature([], tf.string),
        'RGB_image/data': tf.io.FixedLenFeature([], tf.string),        
        "site": tf.io.FixedLenFeature([], tf.int64),        
        "elevation": tf.io.FixedLenFeature([], tf.int64),        
        "HSI_image/height": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/width": tf.io.FixedLenFeature([], tf.int64),
        "HSI_image/depth": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/height": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/width": tf.io.FixedLenFeature([], tf.int64),
        "RGB_image/depth": tf.io.FixedLenFeature([], tf.int64),        
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "height": tf.io.FixedLenFeature([], tf.float32),        
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),        
        'box_index': tf.io.FixedLenFeature([], tf.string)
    }

    # Load one example and parse
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_height = tf.cast(example['HSI_image/height'], tf.int64)
    HSI_width = tf.cast(example['HSI_image/width'], tf.int64)
    HSI_depth = tf.cast(example['HSI_image/depth'], tf.int64)
    HSI_image = tf.io.decode_raw(example['HSI_image/data'], tf.uint16)
    HSI_image_shape = tf.stack([HSI_height,HSI_width, HSI_depth])
    
    # Reshape to known shape
    loaded_HSI_image = tf.reshape(HSI_image, HSI_image_shape, name="cast_loaded_HSI_image")
    loaded_HSI_image = tf.cast(loaded_HSI_image, dtype=tf.float32)
    
    # Load RGB image from file
    RGB_height = tf.cast(example['RGB_image/height'], tf.int64)
    RGB_width = tf.cast(example['RGB_image/width'], tf.int64)
    RGB_depth = tf.cast(example['RGB_image/depth'], tf.int64)
    RGB_image = tf.io.decode_raw(example['RGB_image/data'], tf.uint16)
    RGB_image_shape = tf.stack([RGB_height,RGB_width, RGB_depth])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(RGB_image, RGB_image_shape, name="cast_loaded_RGB_image")
    loaded_RGB_image = tf.cast(loaded_RGB_image, dtype=tf.float32)
    
    #Metadata and labels
    
    #recast and scale to km    
    number_of_sites = tf.cast(example['number_of_sites'], tf.int32)        
    site = tf.cast(example['site'], tf.int64)    
    elevation = tf.cast(example['elevation'], tf.float32)    
    elevation = elevation / 1000
    metadata = elevation

    return (loaded_HSI_image, loaded_RGB_image), example["box_index"]


def _metadata_parse_(tfrecord):
    """Tfrecord generator parse for a metadata model only"""
    # Define features
    features = {
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),  
        "elevation": tf.io.FixedLenFeature([], tf.int64),          
        "classes": tf.io.FixedLenFeature([], tf.int64),       
        "height": tf.io.FixedLenFeature([], tf.float32),
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64)                        
    }

    example = tf.io.parse_single_example(tfrecord, features)
    height = tf.cast(example['height'], tf.float32)

    site = tf.cast(example['site'], tf.int64)
    sites = tf.cast(example['number_of_sites'], tf.int32)    
    
    elevation = tf.cast(example['elevation'], tf.int64)
    elevation = elevation/1000
    height = height / 100
    
    
    label = tf.cast(example['label'], tf.int64)
    classes = tf.cast(example['classes'], tf.int32)    

    #one hot
    one_hot_labels = tf.one_hot(label, classes)
    one_hot_sites = tf.one_hot(site, sites)

    return (elevation, height, one_hot_sites), one_hot_labels

def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def preproccess_images(data):
    """Ensemble preprocessing, assume HSI, RGB, Metadata order in data"""
    HSI, RGB, elevation, height, site = data 
    HSI = tf.image.per_image_standardization(HSI)
    #RGB = tf.image.per_image_standardization(RGB)
    HSI = flip(HSI)
    RGB = flip(RGB)
    
    #Rotate
    HSI = tf.image.rot90(HSI)
    RGB = tf.image.rot90(RGB)
    
    return HSI, RGB, elevation, height, site

def tf_dataset(tfrecords,
               batch_size=2,
               shuffle=True,
               mode="HSI_train",
               normalize=True,
               drop_remainder=False,
               cores=32):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        tfrecords: path to tfrecords, see generate.py
        mode:  "train" mode records include training labels, "submodel" triples the layers to match number of softmax layers,  "predict" is just image data and coordinates
        normalize: Whether to normalize RGB data. 
    Returns:
        dataset: a tf.data dataset yielding crops and labels for train: True, crops and raster indices for train: False
        """
    AUTO = tf.data.experimental.AUTOTUNE
    ignore_order = tf.data.Options()

    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=cores)
    dataset = dataset.with_options(ignore_order)

    if shuffle:
        print("Shuffling data")
        dataset = dataset.shuffle(buffer_size=10)

    if mode == "HSI_train":
        dataset = dataset.map(_HSI_train_parse_, num_parallel_calls=cores)
        #normalize and batch
        dataset = dataset.map(lambda inputs, label: (tf.image.per_image_standardization(inputs), label))
        dataset = dataset.map(lambda inputs, label: (flip(inputs), label))   
        dataset = dataset.map(lambda image, label: (tf.image.rot90(image), label))                        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    
    elif mode == "RGB_train":
        dataset = dataset.map(_RGB_train_parse_, num_parallel_calls=cores)
        #normalize and batch
        #dataset = dataset.map(lambda inputs, label: (tf.image.per_image_standardization(inputs), label))
        dataset = dataset.map(lambda inputs, label: (flip(inputs), label))        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 2)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
     
    elif mode == "ensemble":
        dataset = dataset.map(_train_parse_, num_parallel_calls=cores)
        #normalize and batch
        dataset = dataset.map(lambda data, label: (preproccess_images(data),label))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
            
    elif mode == "predict":
        dataset = dataset.map(_predict_parse_, num_parallel_calls=cores)
        dataset = dataset.map(lambda inputs, index: ((tf.image.per_image_standardization(inputs[0]),inputs[1]), index))
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    
    elif mode == "metadata":
        dataset = dataset.map(_metadata_parse_, num_parallel_calls=cores)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        
    elif mode == "HSI_submodel":
        dataset = dataset.map(_train_HSI_submodel_parse_, num_parallel_calls=cores)
        dataset = dataset.map(lambda image, label: (tf.image.per_image_standardization(image), label))   
        dataset = dataset.map(lambda image, label: (flip(image), label))     
        dataset = dataset.map(lambda image, label: (tf.image.rot90(image), label))                
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    elif mode == "RGB_submodel":
        dataset = dataset.map(_train_RGB_submodel_parse_, num_parallel_calls=cores)
        #dataset = dataset.map(lambda image, label: (tf.image.per_image_standardization(image), label))   
        dataset = dataset.map(lambda image, label: (flip(image), label))        
        if shuffle:
            dataset = dataset.shuffle(buffer_size=batch_size * 2)
        dataset = dataset.batch(batch_size=batch_size, drop_remainder=drop_remainder)     
    else:
        raise ValueError(
            "invalid mode, please use HSI_train, RGB_train, ensemble, predict or submodel: {}".format(mode))
    
    dataset = dataset.prefetch(buffer_size=1)

    return dataset
