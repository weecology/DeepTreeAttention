#### tf.data input pipeline ###
import geopandas as gpd
import numpy as np
import os
import pandas as pd
import rasterio
import random
import tensorflow as tf
import cv2
import math

from rasterio.windows import from_bounds
from DeepTreeAttention.generators import neighbors

from shapely import wkt

def image_normalize(image):
    """normalize a 3d numoy array simiiar to tf.image.per_image_standardization"""
    mean = image.mean()
    stddev = image.std()
    adjusted_stddev = max(stddev, 1.0/math.sqrt(image.size))
    standardized_image = (image - mean) / adjusted_stddev
    
    return standardized_image

def resize(img, height, width):
    # resize image
    dim = (width, height)
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
            expanded_left = left - expand_width
            expanded_bottom = bottom - expand_height
            expanded_right = right + expand_width
            expanded_top =  top+expand_height
        else:
            #Make sure of no negative boxes
            expanded_left = left+expand_width
            expanded_bottom = bottom+expand
            expanded_right = right-expand_width
            expanded_top =  top-expand_height            
        
        window = rasterio.windows.from_bounds(expanded_left, expanded_bottom, expanded_right, expanded_top, transform=src.transform)
        masked_image = src.read(window=window)
    except Exception as e:
        raise ValueError("sensor path: {} failed at reading window {} with error {}".format(src, box.bounds,e))
        
    #Roll depth to channel last
    masked_image = np.rollaxis(masked_image, 0, 3)
    
    #Skip empty frames
    if masked_image.size ==0:
        raise ValueError("Empty frame crop for box {} in sensor path {}".format(box, src))
        
    return masked_image
    
def generate_tfrecords(
                       HSI_sensor_path,
                       RGB_sensor_path,
                       domain,
                       site,
                       number_of_sites,
                       number_of_domains,                       
                       elevation,
                       heights,
                       species_label_dict,
                       chunk_size=1000,
                       savedir=".",
                       HSI_size=20,
                       RGB_size=100,
                       classes=20,
                       train=True,
                       extend_HSI_box=0,
                       extend_RGB_box=0,
                       shuffle=True,
                       shapefile=None,
                       csv_file=None,
                       label_column="label",
                       ensemble_model=None,
                       k_neighbors=5):
    """Yield one instance of data with one hot labels
    Args:
        chunk_size: number of windows per tfrecord
        savedir: directory to save tfrecords
        domain: metadata site domain as integer
        site: metadata site label as integer
        elevation: height above sea level in meters
        heights: height in m
        label_dict: taxonID -> numeric label
        RGB_size: size in pixels of one side of image
        HSI_size: size in pixels of one side of image
        train: training mode to include yielded labels
        number_of_sites: total number of sites used for one-hot encoding
        extend_HSI_box: units in meters to expand DeepForest bounding box to give crop more context
        extend_RGB_box: units in meters to expand DeepForest bounding box to give crop more context
        include_neighbors: logical, whether to extract HSI data from neighbor trees.
        ensemble_model: an ensemble model that predicts neighbor features
        k_neighbors: number of neighbors to extract

    Returns:
        filename: tfrecords path
    """
    
    if all([x is None for x in [csv_file, shapefile]]):
        raise AttributeError("Either pass a shapefile=, or csv_file argument")
    
    HSI_src = rasterio.open(HSI_sensor_path)
    RGB_src = rasterio.open(RGB_sensor_path)
    
    #Read csv file
    if shapefile is None:
        basename = os.path.splitext(os.path.basename(csv_file))[0]        
        gdf = pd.read_csv(csv_file)
        gdf['geometry'] = gdf['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(gdf)
        #assign crs
        gdf.crs = RGB_src.crs
                
    else:
        basename = os.path.splitext(os.path.basename(shapefile))[0]        
        gdf = gpd.read_file(shapefile)
    
    #Remove any nan and species not in the label dict if provided
    gdf = gdf[~gdf[label_column].isnull()]
    if species_label_dict is not None:
        gdf = gdf[gdf[label_column].isin(list(species_label_dict.keys()))]
    
    gdf["box_index"] = gdf.index.values
    labels = []
    HSI_crops = []
    RGB_crops = []
    indices = []
    neighbor_arrays = []
    neighbor_distances = []
    
    #Give an individual column
    gdf["individual"] = gdf.index.values
    
    for index, row in gdf.iterrows():
        #Add training label, ignore unclassified 0 class
        if train:
            labels.append(row[label_column])
        try:
            HSI_crop = crop_image(HSI_src, row["geometry"], extend_HSI_box)
            RGB_crop = crop_image(RGB_src, row["geometry"], extend_RGB_box)
        except Exception as e:
            print("row {} failed with {}".format(index, e))
            continue
        
        HSI_crops.append(HSI_crop)
        RGB_crops.append(RGB_crop)
        indices.append(int(row["box_index"]))
    
        #extract neighbors
        #Encode metadata
        one_hot_sites = tf.one_hot(site, number_of_sites)  
        one_hot_domains = tf.one_hot(domain, number_of_domains)
        metadata = [elevation, one_hot_sites, one_hot_domains]
        
        neighbor_pool = gdf[~(gdf.individual == row["individual"])].reset_index(drop=True)
        raster = rasterio.open(HSI_sensor_path)
        neighbor_array, neighbor_distance = neighbors.predict_neighbors(row, metadata=metadata, HSI_size=HSI_size, raster=raster, neighbor_pool=neighbor_pool, model=ensemble_model, k_neighbors=k_neighbors)
        neighbor_arrays.append(neighbor_array)
        neighbor_distances.append(neighbor_distance)
            
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
            z = list(zip(HSI_crops, RGB_crops, heights, indices, numeric_species_labels, neighbor_arrays, neighbor_distances))
            random.shuffle(z)
            HSI_crops, RGB_crops, heights, indices, numeric_species_labels, neighbor_arrays, neighbor_distances = zip(*z)

    #get keys and divide into chunks for a single tfrecord
    filenames = []
    counter = 0
    for i in range(0, len(HSI_crops) + 1, chunk_size):
        chunk_HSI_crops = HSI_crops[i:i + chunk_size]
        chunk_RGB_crops = RGB_crops[i:i + chunk_size]        
        chunk_index = indices[i:i + chunk_size]
        chunk_height = heights[i:i + chunk_size]
        chunk_neighbor_arrays = neighbor_arrays[i:i + chunk_size]
        chunk_neighbor_distances = neighbor_distances[i:i + chunk_size]
        
        #All records in a single shapefile are the same site
        chunk_sites = np.repeat(site, len(chunk_index))
        chunk_domains = np.repeat(domain, len(chunk_index))
        chunk_elevations = np.repeat(elevation, len(chunk_index))
        
        if train:
            chunk_labels = numeric_species_labels[i:i + chunk_size]
        else:
            chunk_labels = None

        #resize crops
        resized_HSI_crops = [resize(x, HSI_size, HSI_size).astype(np.float32) for x in chunk_HSI_crops]
        resized_RGB_crops = [resize(x, RGB_size, RGB_size).astype(np.float32) for x in chunk_RGB_crops]
        
        resized_HSI_crops = [image_normalize(x) for x in resized_HSI_crops]

        filename = "{}/{}_{}.tfrecord".format(savedir, basename, counter)
        
        write_tfrecord(filename=filename,
                       HSI_images=resized_HSI_crops,
                       RGB_images=resized_RGB_crops,
                       labels=chunk_labels,
                       domains = chunk_domains,
                       sites=chunk_sites,
                       heights=chunk_height,
                       elevations= chunk_elevations,
                       indices=chunk_index,
                       neighbor_arrays=chunk_neighbor_arrays,
                       neighbor_distances=chunk_neighbor_distances,
                       number_of_sites=number_of_sites,
                       number_of_domains=number_of_domains,     
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

def write_tfrecord(filename, HSI_images, RGB_images, domains, sites, elevations, heights, indices, number_of_domains, number_of_sites, classes, neighbor_arrays, neighbor_distances, labels=None):
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
                domain=domains[index],
                site = sites[index],
                HSI_image = HSI_images[index],
                RGB_image = RGB_images[index],
                label=labels[index],
                height=heights[index],                
                elevation=elevations[index],
                number_of_sites=number_of_sites,
                number_of_domains=number_of_domains,   
                neighbor_arrays=neighbor_arrays[index],
                neighbor_distances=neighbor_distances[index],
                classes=classes)
            writer.write(tf_example.SerializeToString())
    else:
        for index, image in enumerate(HSI_images):
            tf_example = create_record(
                index=indices[index],
                domain = domains[index],
                site = sites[index],
                elevation = elevations[index],
                HSI_image=image,
                height=heights[index],
                RGB_image = RGB_images[index],
                number_of_sites=number_of_sites,
                number_of_domains=number_of_domains,   
                neighbor_arrays=neighbor_arrays[index],
                neighbor_distances=neighbor_distances[index],                
                classes=classes)
            writer.write(tf_example.SerializeToString())

    writer.close()


def create_record(HSI_image, RGB_image, index, domain, site, elevation, height, classes, number_of_sites,number_of_domains, neighbor_arrays=None, neighbor_distances=None, label=None):
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
    
    #Standardize HSI normalization, perform now instead of at runtime.
    HSI_rows = HSI_image.shape[0]
    HSI_cols = HSI_image.shape[1]
    HSI_depth = HSI_image.shape[2]
    
    RGB_rows = RGB_image.shape[0]
    RGB_cols = RGB_image.shape[1]
    RGB_depth = RGB_image.shape[2]

    feature={
        'box_index': _int64_feature(index),
        'HSI_image/data': tf.train.Feature(float_list=tf.train.FloatList(value=HSI_image.reshape(-1))),
        'domain': _int64_feature(domain),                    
        'site': _int64_feature(site),    
        'elevation': _float32_feature(elevation),                                
        'HSI_image/height': _int64_feature(HSI_rows),
        'HSI_image/width': _int64_feature(HSI_cols),
        'HSI_image/depth': _int64_feature(HSI_depth),
        'RGB_image/data': tf.train.Feature(float_list=tf.train.FloatList(value=RGB_image.reshape(-1))),                            
        'RGB_image/height': _int64_feature(RGB_rows),
        'RGB_image/width': _int64_feature(RGB_cols),
        'RGB_image/depth': _int64_feature(RGB_depth),                
        'classes': _int64_feature(classes),                
        'number_of_domains': _int64_feature(number_of_domains),                
        'number_of_sites': _int64_feature(number_of_sites),
        'height': _float32_feature(height)
    }
    
    if label is not None:
        feature["label"] = _int64_feature(label)
    
    if neighbor_arrays is not None:
        #all arrays should have same shape.
        feature["k_neighbors"] = _int64_feature(neighbor_arrays.shape[0])
        feature["n_neighbor_features"] = _int64_feature(neighbor_arrays.shape[1])
        feature["neighbor_arrays"] = _bytes_feature(neighbor_arrays.tobytes())
        feature["neighbor_distances"] = tf.train.Feature(float_list=tf.train.FloatList(value=neighbor_distances))
        
    # Serialize to string and write to file
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    return example

#TODO, I need a seperate ensemble parser for training the ensemble before neighbors and one afterwards. Also neighbor distances needs to be size encoded.
def _ensemble_parse_(tfrecord):
    features = {
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),
        "site": tf.io.FixedLenFeature([], tf.int64),  
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64),  
        "height": tf.io.FixedLenFeature([], tf.float32),     
        "domain": tf.io.FixedLenFeature([], tf.int64),  
        "number_of_domains": tf.io.FixedLenFeature([], tf.int64),           
        "elevation": tf.io.FixedLenFeature([], tf.float32),   
        "k_neighbors": tf.io.FixedLenFeature([],tf.int64),
        "n_neighbor_features": tf.io.FixedLenFeature([],tf.int64),
        'neighbor_arrays' : tf.io.FixedLenFeature([], tf.string),   
        'neighbor_distances': tf.io.FixedLenFeature([5],tf.float32)
    }
    
    features['HSI_image/data'] = tf.io.FixedLenFeature([20*20*369], tf.float32)        
    features["HSI_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/depth"] = tf.io.FixedLenFeature([], tf.int64)
          
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load HSI image from file
    HSI_image_shape = tf.stack([example['HSI_image/height'],example['HSI_image/width'], example['HSI_image/depth']])
    
    # Reshape to known shape
    loaded_HSI_image = tf.reshape(example['HSI_image/data'], HSI_image_shape, name="cast_loaded_HSI_image")
    
    ## Parse and reshape neighbor matrix
    flat_neighbor_arrays = tf.io.decode_raw(example["neighbor_arrays"], tf.float32)
    neighbor_array_shape = tf.stack([example["k_neighbors"], example["n_neighbor_features"]])
    
    neighbor_arrays = tf.reshape(flat_neighbor_arrays, neighbor_array_shape)
    
    site = example['site']
    sites = tf.cast(example['number_of_sites'], tf.int32)    
    
    #one hot
    one_hot_sites = tf.one_hot(site, sites)
    
    #labels
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    domain = example['domain']
    domains = tf.cast(example['number_of_domains'], tf.int32)    
    one_hot_domains = tf.one_hot(domain, domains)
    
    return (loaded_HSI_image, neighbor_arrays, example['elevation'], one_hot_sites, one_hot_domains), one_hot_labels

def _HSI_parse_(tfrecord):
    features = {
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),   
    }
    
    features['HSI_image/data'] = tf.io.FixedLenFeature([20*20*369], tf.float32)        
    features["HSI_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/depth"] = tf.io.FixedLenFeature([], tf.int64)
    
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_image_shape = tf.stack([example['HSI_image/height'],example['HSI_image/width'], example['HSI_image/depth']])
    loaded_HSI_image = tf.reshape(example['HSI_image/data'], HSI_image_shape, name="cast_loaded_HSI_image")
        
    #labels
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    return loaded_HSI_image, one_hot_labels

def _HSI_autoencoder_parse_(tfrecord):
    features = {
    }
    
    features['HSI_image/data'] = tf.io.FixedLenFeature([20*20*369], tf.float32)        
    features["HSI_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/depth"] = tf.io.FixedLenFeature([], tf.int64)
    
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_image_shape = tf.stack([example['HSI_image/height'],example['HSI_image/width'], example['HSI_image/depth']])
    loaded_HSI_image = tf.reshape(example['HSI_image/data'], HSI_image_shape, name="cast_loaded_HSI_image")
    
    return loaded_HSI_image, loaded_HSI_image

def _HSI_submodel_parse_(tfrecord):
    features = {
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),   
    }
    
    features['HSI_image/data'] = tf.io.FixedLenFeature([20*20*369], tf.float32)        
    features["HSI_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["HSI_image/depth"] = tf.io.FixedLenFeature([], tf.int64)
    
    example = tf.io.parse_single_example(tfrecord, features)

    # Load HSI image from file
    HSI_image_shape = tf.stack([example['HSI_image/height'],example['HSI_image/width'], example['HSI_image/depth']])
    loaded_HSI_image = tf.reshape(example['HSI_image/data'], HSI_image_shape, name="cast_loaded_HSI_image")
        
    #labels
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    return loaded_HSI_image, (one_hot_labels,one_hot_labels,one_hot_labels)

def _RGB_parse_(tfrecord):
    features = {
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),  
    }
    
    features['RGB_image/data'] = tf.io.FixedLenFeature([100*100*3], tf.float32)        
    features["RGB_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["RGB_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["RGB_image/depth"] = tf.io.FixedLenFeature([], tf.int64)             
    
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load RGB image from file
    RGB_image_shape = tf.stack([example['RGB_image/height'],example['RGB_image/width'], example['RGB_image/depth']])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(example['RGB_image/data'], RGB_image_shape, name="cast_loaded_RGB_image")
    
    #labels
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    return loaded_RGB_image, one_hot_labels

def _RGB_submodel_parse_(tfrecord):
    features = {
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64),  
    }
    
    features['RGB_image/data'] = tf.io.FixedLenFeature([100*100*3], tf.float32)        
    features["RGB_image/height"] =  tf.io.FixedLenFeature([], tf.int64)
    features["RGB_image/width"] = tf.io.FixedLenFeature([], tf.int64)
    features["RGB_image/depth"] = tf.io.FixedLenFeature([], tf.int64)             
    
    example = tf.io.parse_single_example(tfrecord, features)
    
    # Load RGB image from file
    RGB_image_shape = tf.stack([example['RGB_image/height'],example['RGB_image/width'], example['RGB_image/depth']])
    
    # Reshape to known shape
    loaded_RGB_image = tf.reshape(example['RGB_image/data'], RGB_image_shape, name="cast_loaded_RGB_image")
    
    #labels
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    return loaded_RGB_image, (one_hot_labels,one_hot_labels,one_hot_labels)

def _box_index_parse_(tfrecord):
    features = {
        'box_index': tf.io.FixedLenFeature([], tf.int64), 
    }
    example = tf.io.parse_single_example(tfrecord, features)
    
    return example["box_index"]
    
def _metadata_parse_(tfrecord):
    """Tfrecord generator parse for a metadata model only"""
    # Define features
    features = {
        "site": tf.io.FixedLenFeature([], tf.int64),  
        "number_of_sites": tf.io.FixedLenFeature([], tf.int64), 
        "domain": tf.io.FixedLenFeature([], tf.int64),  
        "number_of_domains": tf.io.FixedLenFeature([], tf.int64),          
        "height": tf.io.FixedLenFeature([], tf.float32),     
        "elevation": tf.io.FixedLenFeature([], tf.float32),  
        "classes": tf.io.FixedLenFeature([], tf.int64),
        "label": tf.io.FixedLenFeature([], tf.int64)   
    }

    example = tf.io.parse_single_example(tfrecord, features)
    
    #One hot site
    site = example['site']
    sites = tf.cast(example['number_of_sites'], tf.int32)    
    one_hot_sites = tf.one_hot(site, sites)
    
    domain = example['domain']
    domains = tf.cast(example['number_of_domains'], tf.int32)    
    one_hot_domains = tf.one_hot(domain, domains)
    
    #one hot elevation
    classes = tf.cast(example['classes'], tf.int32)    
    one_hot_labels = tf.one_hot(example['label'], classes)
    
    return (example['elevation'], one_hot_sites, one_hot_domains), one_hot_labels

def augment(data, label):
    """Ensemble preprocessing, assume HSI, RGB, Metadata order in data"""

    data = tf.image.rot90(data)
    data = tf.image.random_flip_left_right(data)
    data = tf.image.random_flip_up_down(data)    
    
    return data, label

def ensemble_augment(data, label):
    """Ensemble preprocessing, assume HSI, RGB, Metadata order in data"""
    
    HSI, neighbor_array, elevation, site, domain = data
    
    HSI = tf.image.rot90(HSI)
    HSI = tf.image.random_flip_left_right(HSI)
    HSI = tf.image.random_flip_up_down(HSI)    

    data = HSI,neighbor_array, elevation, site, domain
    
    return data, label

def normalize(data):
    data = tf.image.per_image_standardization(data)
    
    return data

def tf_dataset(tfrecords,
               batch_size=2,
               shuffle=True,
               mode = "ensemble",
               ids = False,
               augmentation = True,
               cache=False,
               cores=32):
    """Create a tf.data dataset that yields sensor data and ground truth
    Args:
        tfrecords: path to tfrecords, see generate.py
        RGB: Include RGB data
        HSI: Include HSI data
        ids: include box ids
        metadata: include metadata 
        labels: training record labels
        submodel: Logical. "spectral" or "spatial submodels" have three label inputs
    Returns:
        dataset: a tf.data dataset yielding crops and labels for train: True, crops and raster indices for train: False
        """
    AUTO = tf.data.experimental.AUTOTUNE
    #For the moment be explicit.
    
    dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=cores)     
    
    #batch and shuffle
    if shuffle:
        dataset = dataset.shuffle(buffer_size=100)
        
    if mode == "ensemble":
        dataset = dataset.map(_ensemble_parse_, num_parallel_calls=cores)
        if cache:
            dataset = dataset.cache()        
        if augmentation:
            dataset = dataset.map(ensemble_augment, num_parallel_calls=cores)        
    elif mode == "HSI_autoencoder":
        dataset = dataset.map(_HSI_autoencoder_parse_)
        if cache:
            dataset = dataset.cache()            
    elif mode == "HSI":
        dataset = dataset.map(_HSI_parse_)
        if cache:
            dataset = dataset.cache()        
        if augmentation:
            dataset = dataset.map(augment, num_parallel_calls=cores)                
    elif mode == "HSI_submodel":
        dataset = dataset.map(_HSI_submodel_parse_)
        if cache:
            dataset = dataset.cache()        
        if augmentation:
            dataset = dataset.map(augment, num_parallel_calls=cores)                
    elif mode == "RGB":
        dataset = dataset.map(_RGB_parse_)
        if augmentation:
            dataset = dataset.map(augment, num_parallel_calls=cores)                
    elif mode == "RGB_submodel":
        dataset = dataset.map(_RGB_submodel_parse_)        
        if augmentation:
            dataset = dataset.map(augment, num_parallel_calls=cores)                
    elif mode == "metadata":
        dataset = dataset.map(_metadata_parse_)
    else:
        raise ValueError("Accepted types = 'ensemble', 'HSI', 'HSI_submodel', 'RGB', 'RGB_submodel', 'metadata'")   
                        
    if ids:
        ids_dataset = tf.data.TFRecordDataset(tfrecords, num_parallel_reads=cores)     
        ids_dataset = ids_dataset.map(_box_index_parse_)
        dataset = tf.data.Dataset.zip((ids_dataset, dataset))  
        
    dataset = dataset.batch(batch_size=batch_size)
    
    dataset = dataset.prefetch(buffer_size=1)    
    
    return dataset