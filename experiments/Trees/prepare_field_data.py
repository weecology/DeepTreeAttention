#Convert NEON field sample points into bounding boxes of cropped image data for model training
import cv2
import os
import glob
import sys
import geopandas as gpd
import rasterio
import random
import numpy as np
import shapely
import pandas as pd
import traceback
from matplotlib import pyplot

from DeepTreeAttention.generators.boxes import write_tfrecord
from DeepTreeAttention.utils.paths import find_sensor_path, convert_h5
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.utils import start_cluster
from DeepTreeAttention.generators import create_training_shp
from distributed import wait
from random import randint
from time import sleep

def resize(img, height, width):
    # resize image
    dim = (width, height)    
    img = img.astype("float32")
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_NEAREST)
    
    return resized

def predict_trees(deepforest_model, rgb_path, bounds, expand=10):
    """Predict an rgb path at specific utm bounds
    Args:
        deepforest_model: a deepforest model object used for prediction
        rgb_path: full path to image
        bounds: utm extent given by geopandas.total_bounds
        expand: numeric meters to add to edges to reduce edge effects
        """
    #DeepForest is trained on 400m crops, easiest to mantain this approximate size centered on points
    left, bottom, right, top = bounds
    expand_width = (40 - (right - left))/2
    left = left - expand_width
    right = right + expand_width
    
    expand_height = (40 - (top - bottom))/2 
    bottom = bottom - expand_height
    top = top + expand_height 
    
    src = rasterio.open(rgb_path)
    pixelSizeX, pixelSizeY  = src.res    
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    src.close()
    
    #roll to bgr channel order, bgr
    img = np.rollaxis(img, 0,3)
    img = img[:,:,::-1]
    
    #reshape to 400x400m
    resized = resize(img, 400, 400)
    boxes = deepforest_model.predict_image(numpy_image = resized, return_plot=False)
    
    if boxes.empty:
        return boxes
    
    #tranform boxes to original size
    x_scale = 400/img.shape[0]
    y_scale = 400/img.shape[1]
    
    boxes["xmin"] = boxes["xmin"]/x_scale 
    boxes["xmax"] = boxes["xmax"]/x_scale 
    boxes["ymin"] = boxes["ymin"]/y_scale 
    boxes["ymax"] = boxes["ymax"]/y_scale     

    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["xmin"] = (boxes["xmin"] *pixelSizeX) + left
    boxes["xmax"] = (boxes["xmax"] * pixelSizeX) + left
    boxes["ymin"] = top - (boxes["ymin"] * pixelSizeY) 
    boxes["ymax"] = top - (boxes["ymax"] * pixelSizeY)

    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    boxes = gpd.GeoDataFrame(boxes, geometry='geometry')    
    
    #Buffer slightly 
    boxes.geometry = boxes.geometry.buffer(1)
    return boxes

def choose_box(group, plot_data):
    """Given a set of overlapping bounding boxes and predictions, just choose a closest to stem box by centroid if there are multiples"""
    if group.shape[0] == 1:
        return  group
    else:
        #Find centroid
        individual_id = group.individual.unique()[0]
        stem_location = plot_data[plot_data["individual"]==individual_id].geometry.iloc[0]
        closest_stem = group.centroid.distance(stem_location).sort_values().index[0]
        return group.loc[[closest_stem]]

def create_boxes(plot_data, size=2):
    """If there are no deepforest boxes, fall back on selecting a fixed area around stem point"""
    fixed_boxes = plot_data.buffer(size).envelope
    
    fixed_boxes = gpd.GeoDataFrame(geometry=fixed_boxes)
    
    #Mimic the existing structure
    fixed_boxes = gpd.sjoin(fixed_boxes, plot_data)
    fixed_boxes["score"] = None
    fixed_boxes["label"] = "Tree" 
    fixed_boxes["xmin"] = None 
    fixed_boxes["xmax"] = None
    fixed_boxes["ymax"] = None
    fixed_boxes["ymin"] = None
    
    return fixed_boxes
    
def process_plot(plot_data, rgb_pool, deepforest_model):
    """For a given NEON plot, find the correct sensor data, predict trees and associate bounding boxes with field data
    Args:
        plot_data: geopandas dataframe in a utm projection
        deepforest_model: deepforest model used for prediction
    Returns:
        merged_boxes: geodataframe of bounding box predictions with species labels
    """
    #DeepForest prediction
    try:
        rgb_sensor_path = find_sensor_path(bounds=plot_data.total_bounds, lookup_pool=rgb_pool)
    except Exception as e:
        raise ValueError("cannot find RGB sensor for {}".format(plot_data.plotID.unique()))
    
    boxes = predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_sensor_path, bounds=plot_data.total_bounds)

    if boxes.empty:
        raise ValueError("No trees predicted in plot: {}, skipping.".format(plot_data.plotID.unique()[0]))
        
    #Merge results with field data, buffer on edge 
    merged_boxes = gpd.sjoin(boxes, plot_data)
    
    #If no remaining boxes just take a box around center
    missing_ids = plot_data[~plot_data.individual.isin(merged_boxes.individual)]
    
    if not missing_ids.empty:
        created_boxes= create_boxes(missing_ids)
        merged_boxes = merged_boxes.append(created_boxes)
        
    #If there are multiple boxes, take the center box
    grouped = merged_boxes.groupby("individual")
    
    cleaned_boxes = []
    for value, group in grouped:
        choosen_box = choose_box(group, plot_data)
        cleaned_boxes.append(choosen_box)
    
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_boxes),crs=merged_boxes.crs)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
    assert plot_data.shape[0] == merged_boxes.shape[0]
    return merged_boxes

def crop_image(sensor_path, box, expand=0): 
    """Read sensor data and crop a bounding box
    Args:
        sensor_path: full path to sensor data
        box: geopandas geometry polygon object
        expand: add padding in meters to the edge of the crop
    Returns:
        masked_image: a crop of sensor data at specified bounds
    """
    #Read data and mask
    try:    
        src = rasterio.open(sensor_path)
        left, bottom, right, top = box.bounds
        window=rasterio.windows.from_bounds(left-expand, bottom-expand, right+expand, top+expand, transform=src.transform)
        masked_image = src.read(window=window)
        src.close()
    except Exception as e:
        raise ValueError("sensor path: {} failed at reading crop window {} with error {}".format(sensor_path, box.bounds,e))
        
    #Roll depth to channel last
    masked_image = np.rollaxis(masked_image, 0, 3)
    
    #Skip empty frames
    if masked_image.size ==0:
        raise ValueError("Empty frame crop for box {} in sensor path {}".format(box, sensor_path))
    
    return masked_image

def create_crops(merged_boxes, hyperspectral_pool=None, rgb_pool=None, sensor="hyperspectral", expand=0, hyperspectral_savedir="."):
    """Crop sensor data based on a dataframe of geopandas bounding boxes
    Args:
        merged_boxes: geopandas dataframe with bounding box geometry, plotID, siteID, and species label
        hyperspectral_pool: glob string for looking up matching sensor tiles
        expand: units in meters to add to crops to give context around deepforest box
        hyperspectral_savedir: location to save convert .tif from .h5 files
    Returns:
        crops: list of cropped sensor data
        labels: species id labels
        box_index: unique index and plot_data length.
    """    
    crops = []
    labels = []
    sites = []
    box_index = []
    elevations = []
    heights = []
    for index, row in merged_boxes.iterrows():
        #Crop and append
        box = row["geometry"]       
        plot_name = row["plotID"] 
        site = row["plotID"].split("_")[0]
        elevation = int(row["elevation"])
        height = row["height"]
        
        #get sensor data
        if sensor == "rgb":
            try:
                sensor_path = find_sensor_path(bounds=box.bounds, lookup_pool=rgb_pool)
            except:
                raise ValueError("Cannot find RGB data path for box bounds {} for plot_name {}".format(box.bounds,plot_name))
        elif sensor == "hyperspectral":
            try:
                rgb_path = find_sensor_path(bounds=box.bounds, lookup_pool=rgb_pool)
            except:
                raise ValueError("Cannot find RGB data path for box bounds {} for plot_name {}".format(box.bounds,plot_name))
                
            try:
                hyperspectral_h5_path = find_sensor_path(bounds=box.bounds, lookup_pool=hyperspectral_pool)
            except:
                raise ValueError("Cannot find hyperspectral data path for box bounds {} for plot_name {}".format(box.bounds,plot_name))
                
            sensor_path = convert_h5(hyperspectral_h5_path, rgb_path, savedir=hyperspectral_savedir)
        
        crop = crop_image(sensor_path=sensor_path, box=box, expand=expand)
        
        crops.append(crop)
        sites.append(site)
        labels.append(row["taxonID"])
        elevations.append(elevation)
        heights.append(height)
        box_index.append("{}_{}".format(plot_name,index))
        
    return crops, labels, sites, heights, elevations, box_index

def create_records(HSI_crops, RGB_crops, labels, sites, heights, elevations, box_index, savedir, RGB_size, HSI_size, chunk_size=400):
    #get keys and divide into chunks for a single tfrecor
    filenames = []
    counter = 0
    for i in range(0, len(HSI_crops)+1, chunk_size):
        chunk_HSI_crops = HSI_crops[i:i + chunk_size]
        chunk_RGB_crops = RGB_crops[i:i + chunk_size]
        chunk_index = box_index[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]
        chunk_sites = sites[i:i + chunk_size]
        chunk_elevations = elevations[i:i + chunk_size]
        chunk_heights = heights[i:i + chunk_size]
            
        resized_RGB_crops = [resize(x, RGB_size, RGB_size).astype("int16") for x in chunk_RGB_crops]
        resized_HSI_crops = [resize(x, HSI_size, HSI_size).astype("int16") for x in chunk_HSI_crops]
        
        filename = "{}/field_data_{}.tfrecord".format(savedir, counter)
        write_tfrecord(filename=filename,
                                            HSI_images=resized_HSI_crops,
                                            RGB_images=resized_RGB_crops,
                                            labels=chunk_labels,
                                            sites = chunk_sites,
                                            heights = chunk_heights,
                                            elevations=chunk_elevations,
                                            indices=chunk_index,
                                            classes=max(labels)+1)
        
        filenames.append(filename)
        counter +=1    
    
    return filenames

def run(plot, df, rgb_pool=None, hyperspectral_pool=None, extend_HSI_box=0, extend_RGB_box=0, hyperspectral_savedir=".",saved_model=None, deepforest_model=None):
    """wrapper function for dask, see main.py"""
    try:
        from deepforest import deepforest
    
        #create deepforest model
        if deepforest_model is None:
            if saved_model is None:
                deepforest_model = deepforest.deepforest()
                deepforest_model.use_release()
            else:
                deepforest_model = deepforest.deepforest(saved_model=saved_model)
            
        #Filter data and process
        plot_data = df[df.plotID == plot]
        predicted_trees = process_plot(plot_data, rgb_pool, deepforest_model)
        
        #Crop HSI
        plot_HSI_crops, plot_labels, plot_sites, plot_heights, plot_elevations, plot_box_index = create_crops(
            predicted_trees,
            hyperspectral_pool=hyperspectral_pool,
            rgb_pool=rgb_pool,
            sensor="hyperspectral",
            expand=extend_HSI_box,
            hyperspectral_savedir=hyperspectral_savedir)
        
        #Crop RGB, drop repeated elements, leave one for testing
        plot_rgb_crops, plot_rgb_labels, _, _, _, _ = create_crops(
            predicted_trees,
            hyperspectral_pool=hyperspectral_pool,
            rgb_pool=rgb_pool,
            sensor="rgb",
            expand=extend_RGB_box,
            hyperspectral_savedir=hyperspectral_savedir)    
        
        #Assert they are the same
        assert len(plot_rgb_crops) == len(plot_HSI_crops)
        assert plot_labels==plot_rgb_labels
    except Exception as e:
        raise ValueError("Plot {} failed {}".format(plot, e))

    return plot_HSI_crops, plot_rgb_crops, plot_labels, plot_sites, plot_heights, plot_elevations, plot_box_index

def main(
    field_data,
    RGB_size,
    HSI_size,
    rgb_dir, 
    species_classes_file,
    site_classes_file,
    hyperspectral_dir,
    savedir=".", 
    chunk_size=400,
    extend_HSI_box=0, 
    extend_RGB_box=0,
    hyperspectral_savedir=".", 
    saved_model=None, 
    client=None, 
    shuffle=True):
    """Prepare NEON field data into tfrecords
    Args:
        field_data: shp file with location and class of each field collected point
        height: height in meters of the resized training image
        width: width in meters of the resized training image
        savedir: direcory to save completed tfrecords
        extend_HSI_box: units in meters to add to the edge of a predicted box to give more context
        extend_RGB_box: units in meters to add to the edge of a predicted box to give more context
        hyperspectral_savedir: location to save converted .h5 to .tif
        client: dask client object to use
        species_classes_file: optional path to a two column csv file with index and species labels
        site_classes_file: optional path to a two column csv file with index and site labels
        shuffle: shuffle lists before writing
    Returns:
        tfrecords: list of created tfrecords
    """ 
    df = gpd.read_file(field_data)
    plot_names = df.plotID.unique()
    
    hyperspectral_pool = glob.glob(hyperspectral_dir, recursive=True)
    rgb_pool = glob.glob(rgb_dir, recursive=True)
    
    labels = []
    HSI_crops = []
    RGB_crops = []
    sites = []
    box_indexes = []    
    elevations = []
    heights = []
    if client is not None:
        futures = []
        for plot in plot_names:
            future = client.submit(
                run,
                plot=plot,
                df=df,
                rgb_pool=rgb_pool,
                hyperspectral_pool=hyperspectral_pool,
                extend_HSI_box=extend_HSI_box,
                extend_RGB_box=extend_RGB_box,                
                hyperspectral_savedir=hyperspectral_savedir,
                saved_model=saved_model
            )
            futures.append(future)
        
        wait(futures)
        for x in futures:
            try:
                plot_HSI_crops, plot_RGB_crops, plot_labels, plot_sites, plot_heights, plot_elevations, plot_box_index = x.result()
                
                #Append to general plot list
                HSI_crops.extend(plot_HSI_crops)
                RGB_crops.extend(plot_RGB_crops)
                labels.extend(plot_labels)
                sites.extend(plot_sites)           
                heights.extend(plot_heights)
                elevations.extend(plot_elevations)
                box_indexes.extend(plot_box_index)        
            except Exception as e:
                print("Future failed with {}".format(e))      
                traceback.print_exc()
    else:
        from deepforest import deepforest        
        deepforest_model = deepforest.deepforest()
        deepforest_model.use_release()        
        for plot in plot_names:
            try:
                plot_HSI_crops, plot_RGB_crops, plot_labels, plot_sites, plot_heights, plot_elevations, plot_box_index = run(
                    plot=plot,
                    df=df,
                    rgb_pool=rgb_pool,
                    hyperspectral_pool=hyperspectral_pool, 
                    extend_HSI_box=extend_HSI_box,
                    extend_RGB_box=extend_RGB_box,   
                    hyperspectral_savedir=hyperspectral_savedir,
                    saved_model=saved_model,
                    deepforest_model=deepforest_model
                )
            except Exception as e:
                print("Plot failed with {}".format(e))      
                traceback.print_exc()  
                continue
    
            #Append to general plot list
            HSI_crops.extend(plot_HSI_crops)
            RGB_crops.extend(plot_RGB_crops)
            labels.extend(plot_labels)
            sites.extend(plot_sites)    
            heights.extend(plot_heights)                        
            elevations.extend(plot_elevations)
            box_indexes.extend(plot_box_index)
            
    if shuffle:
        z = list(zip(HSI_crops, RGB_crops, sites, heights, elevations, box_indexes, labels))
        random.shuffle(z)
        HSI_crops, RGB_crops, sites, heights, elevations, box_indexes, labels = zip(*z)
                        
    #If passes a species label dict
    species_classdf  = pd.read_csv(species_classes_file)
    species_label_dict = species_classdf.set_index("taxonID").label.to_dict()

    #If passes a site label dict
    if site_classes_file is not None:
        site_classdf  = pd.read_csv(site_classes_file)
        site_label_dict = site_classdf.set_index("siteID").label.to_dict()
    else:
        #Create and save a new site and site label dict
        unique_site_labels = np.unique(sites)
        site_label_dict = {}
        
        for index, label in enumerate(unique_site_labels):
            site_label_dict[label] = index
        pd.DataFrame(site_label_dict.items(), columns=["siteID","label"]).to_csv("{}/site_class_labels.csv".format(savedir))

    #Convert labels to numeric
    numeric_labels = [species_label_dict[x] for x in labels]
    numeric_sites = [site_label_dict[x] for x in sites]
    
    print("Writing records of {} HSI samples, {} RGB samples from {} species and {} sites".format(
        len(HSI_crops),
        len(RGB_crops),
        len(np.unique(numeric_labels)),
        len(np.unique(numeric_sites))))
    
    #Write tfrecords
    tfrecords = create_records(
        HSI_crops=HSI_crops,
        RGB_crops=RGB_crops,
        labels=numeric_labels, 
        sites=numeric_sites, 
        elevations=elevations,
        box_index=box_indexes, 
        savedir=savedir, 
        heights=heights,
        RGB_size=RGB_size,
        HSI_size=HSI_size, 
        chunk_size=chunk_size)
    
    return tfrecords
    
if __name__ == "__main__":
    #Generate the training data shapefiles
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    lookup_glob = "/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif"
    
    #Read config from top level dir
    config = parse_yaml("{}/conf/tree_config.yml".format(ROOT))
    
    #Create train test split
    create_training_shp.train_test_split(ROOT, lookup_glob, n=config["train"]["resampled_per_taxa"])
    
    #create dask client
    client = start_cluster.start(cpus=config["cpu_workers"], mem_size="15GB")
    #client = None
    #test data
    main(
        field_data=config["evaluation"]["ground_truth_path"],
        RGB_size=config["train"]["RGB"]["crop_size"],
        HSI_size=config["train"]["HSI"]["crop_size"],      
        hyperspectral_dir=config["hyperspectral_sensor_pool"],
        rgb_dir=config["rgb_sensor_pool"],
        extend_HSI_box = config["train"]["HSI"]["extend_box"],
        extend_RGB_box = config["train"]["RGB"]["extend_box"],        
        hyperspectral_savedir=config["hyperspectral_tif_dir"],
        savedir=config["evaluation"]["tfrecords"],
        species_classes_file = "{}/data/processed/species_class_labels.csv".format(ROOT),
        site_classes_file =  "{}/data/processed/site_class_labels.csv".format(ROOT),        
        client=client,
        saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    )
    
    print("Evaluation records complete")
    #clean out client of any objects
    client.restart()
    
    ##train data
    main(
        field_data=config["train"]["ground_truth_path"],
        RGB_size=config["train"]["RGB"]["crop_size"],
        HSI_size=config["train"]["HSI"]["crop_size"],        
        hyperspectral_dir=config["hyperspectral_sensor_pool"],
        rgb_dir=config["rgb_sensor_pool"],
        extend_HSI_box = config["train"]["HSI"]["extend_box"],
        extend_RGB_box = config["train"]["RGB"]["extend_box"],     
        hyperspectral_savedir=config["hyperspectral_tif_dir"],
        savedir=config["train"]["tfrecords"],
        client=client,
        species_classes_file = "{}/data/processed/species_class_labels.csv".format(ROOT),
        site_classes_file =  "{}/data/processed/site_class_labels.csv".format(ROOT),        
        saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    )
    
