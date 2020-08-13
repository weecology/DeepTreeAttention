#Convert NEON field sample points into bounding boxes of cropped image data for model training
import cv2
import os
import glob
import geopandas as gpd
import rasterio
import numpy as np
import shapely
import pandas as pd

from DeepTreeAttention.generators.boxes import write_tfrecord
from DeepTreeAttention.utils.paths import find_sensor_path, convert_h5
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.utils import start_cluster

from distributed import wait

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
    left, bottom, right, top = bounds
    src = rasterio.open(rgb_path)
    pixelSizeX, pixelSizeY  = src.res    
    img = src.read(window=rasterio.windows.from_bounds(left-expand, bottom-expand, right+expand, top+expand, transform=src.transform))
    
    #roll to bgr channel order, bgr
    img = np.rollaxis(img, 0,3)
    img = img[:,:,::-1]
    
    boxes = deepforest_model.predict_tile(numpy_image=img, patch_overlap=0.1)

    #subtract origin. Recall that numpy origin is top left! Not bottom left.
    boxes["xmin"] = (boxes["xmin"] *pixelSizeX) + left
    boxes["xmax"] = (boxes["xmax"] * pixelSizeX) + left
    boxes["ymin"] = top - (boxes["ymin"] * pixelSizeY) 
    boxes["ymax"] = top - (boxes["ymax"] * pixelSizeY)

    # combine column to a shapely Box() object, save shapefile
    boxes['geometry'] = boxes.apply(lambda x: shapely.geometry.box(x.xmin,x.ymin,x.xmax,x.ymax), axis=1)
    boxes = gpd.GeoDataFrame(boxes, geometry='geometry')    
    
    return boxes

def process_plot(plot_data, rgb_pool, deepforest_model):
    """For a given NEON plot, find the correct sensor data, predict trees and associate bounding boxes with field data
    Args:
        plot_data: geopandas dataframe in a utm projection
        deepforest_model: deepforest model used for prediction
    Returns:
        merged_boxes: geodataframe of bounding box predictions with species labels
    """
    #DeepForest prediction
    rgb_sensor_path = find_sensor_path(bounds=plot_data.total_bounds, lookup_pool=rgb_pool, sensor="rgb")
    boxes = predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_sensor_path, bounds=plot_data.total_bounds)

    #Merge results with field data
    merged_boxes = gpd.sjoin(boxes, plot_data)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
    return merged_boxes

def crop_image(sensor_path, box, expand=10):
    """Read sensor data and crop a bounding box
    Args:
        sensor_path: full path to sensor data
        box: geopandas geometry polygon object
        expand: add padding in meters to the edge of the crop
    Returns:
        masked_image: a crop of sensor data at specified bounds
    """
    #Read data and mask
    src = rasterio.open(sensor_path)
    left, bottom, right, top = box.bounds
    window=rasterio.windows.from_bounds(left-expand, bottom-expand, right+expand, top+expand, transform=src.transform)
    try:
        masked_image = src.read(window=window)
    except Exception as e:
        raise ValueError("sensor path: {} failed at reading window {} with error {}".format(sensor_path, box.bounds,e))
        
    #Roll depth to channel last
    masked_image = np.rollaxis(masked_image, 0, 3)
    
    #Skip empty frames
    if masked_image.size ==0:
        raise ValueError("Empty frame crop for box {} in sensor path {}".format(box, sensor_path))
    
    return masked_image

def create_crops(merged_boxes, hyperspectral_pool=None, rgb_pool=None, sensor="hyperspectral", expand=0, hyperspectral_savedir="."):
    """Crop sensor data based on a dataframe of geopandas bounding boxes
    Args:
        merged_boxes: geopandas dataframe with bounding box geometry, plotID, and species label
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
    box_index = []
    for index, row in merged_boxes.iterrows():
        #Crop and append
        box = row["geometry"]       
        plot_name = row["plotID"] 
        
        #get sensor data
        if sensor == "rgb":
            sensor_path = find_sensor_path(bounds=box.bounds, lookup_pool=rgb_pool, sensor="rgb")
        elif sensor == "hyperspectral":
            rgb_path = find_sensor_path(bounds=box.bounds, lookup_pool=rgb_pool, sensor="rgb")
            hyperspectral_h5_path = find_sensor_path(bounds=box.bounds, lookup_pool=hyperspectral_pool, sensor="hyperspectral")
            sensor_path = convert_h5(hyperspectral_h5_path, rgb_path, savedir=hyperspectral_savedir)
        
        crop = crop_image(sensor_path=sensor_path, box=box, expand=expand)
        
        crops.append(crop)
        labels.append(row["taxonID"])
        box_index.append("{}_{}".format(plot_name,index))
        
    return crops, labels, box_index

def create_records(crops, labels, box_index, savedir, height, width, chunk_size=200):
    #get keys and divide into chunks for a single tfrecord
    filenames = []
    counter = 0
    for i in range(0, len(crops)+1, chunk_size):
        chunk_crops = crops[i:i + chunk_size]
        chunk_index = box_index[i:i + chunk_size]
        chunk_labels = labels[i:i + chunk_size]

        #resize crops
        resized_crops = [resize(x, height, width).astype("int16") for x in chunk_crops]
        
        filename = "{}/field_data_{}.tfrecord".format(savedir, counter)
        write_tfrecord(filename=filename,
                                            images=resized_crops,
                                            labels=chunk_labels,
                                            indices=chunk_index,
                                            classes=max(labels))
        
        filenames.append(filename)
        counter +=1    
    
    return filenames

def run(plot, df, rgb_pool=None, hyperspectral_pool=None, sensor="hyperspectral", extend_box=0, hyperspectral_savedir="."):
    """wrapper function for dask, see main.py"""
    from deepforest import deepforest
    
    #create deepforest model
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    
    #Filter data and process
    plot_data = df[df.plotID == plot]
    predicted_trees = process_plot(plot_data, rgb_pool, deepforest_model)
    plot_crops, plot_labels, plot_box_index = create_crops(
        predicted_trees,
        hyperspectral_pool=hyperspectral_pool,
        rgb_pool=rgb_pool,
        sensor=sensor,
        expand=extend_box,
        hyperspectral_savedir=hyperspectral_savedir
    )
    
    return plot_crops, plot_labels, plot_box_index

def main(field_data, height, width, rgb_pool=None, hyperspectral_pool=None, sensor="hyperspectral", savedir=".", chunk_size=200, extend_box=0, hyperspectral_savedir=".", n_workers=20):
    """Prepare NEON field data into tfrecords
    Args:
        field_data: shp file with location and class of each field collected point
        height: height in meters of the resized training image
        width: width in meters of the resized training image
        sensor: 'rgb' or 'hyperspecral' image crop
        savedir: direcory to save completed tfrecords
        extend_box: units in meters to add to the edge of a predicted box to give more context
        hyperspectral_savedir: location to save converted .h5 to .tif
        n_workers: number of dask workers
    Returns:
        tfrecords: list of created tfrecords
    """
    #Check sensor type has paths
    if sensor == "hyperspectral":
        assert not hyperspectral_pool is None
    if sensor=="rgb":
        assert not rgb_pool is None
        
    df = gpd.read_file(field_data)
    plot_names = df.plotID.unique()
    
    client = start_cluster.start(cpus=n_workers)
    futures = []
    for plot in plot_names:
        future = client.submit(
            run,
            plot=plot,
            df=df,
            rgb_pool=rgb_pool,
            hyperspectral_pool=hyperspectral_pool,
            sensor=sensor,
            extend_box=extend_box,
            hyperspectral_savedir=hyperspectral_savedir
        )
        futures.append(future)
    
    wait(futures)
    
    labels = []
    crops = []
    box_indexes = []
    for x in futures:
        try:
            plot_crops, plot_labels, plot_box_index = x.result()
            print(plot_box_index[0])
            #Append to general plot list
            crops.extend(plot_crops)
            labels.extend(plot_labels)
            box_indexes.extend(plot_box_index)            
        except Exception as e:
            print("plot {} failed with {}".format(plot,e))

    #Convert labels to numeric
    unique_labels = np.unique(labels)
    label_dict = {}
    
    for index, label in enumerate(unique_labels):
        label_dict[label] = index
        
    numeric_labels = [label_dict[x] for x in labels]
    pd.DataFrame(label_dict.items(), columns=["taxonID","index"]).to_csv("{}/class_labels.csv".format(savedir))
    
    #Write tfrecords
    tfrecords = create_records(crops, numeric_labels, box_indexes, savedir, height, width, chunk_size=chunk_size)
    
    return tfrecords
    
if __name__ == "__main__":
    #Read config 
    config = parse_yaml("../../conf/tree_config.yml")
    main(
        field_data=config["train"]["ground_truth_path"],
        height=config["train"]["crop_size"],
        width=config["train"]["crop_size"],        
        hyperspectral_pool=config["train"]["hyperspectral_sensor_pool"],
        rgb_pool=config["train"]["rgb_sensor_pool"],
        extend_box=config["train"]["extend_box"],
        hyperspectral_savedir=config["hyperspectral_tif_dir"],
        savedir=config["train"]["tfrecords"],
        n_workers=config["cpu_workers"]
    )