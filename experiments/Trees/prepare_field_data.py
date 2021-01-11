#Convert NEON field sample points into bounding boxes of cropped image data for model training
import cv2
import os
import glob
import geopandas as gpd
import rasterio
import numpy as np
import math
import shapely
import pandas as pd
import traceback

from DeepTreeAttention.utils.paths import find_sensor_path, convert_h5
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.utils import start_cluster
from DeepTreeAttention.generators import create_training_shp
from distributed import wait       

def normalize(image):
    """normalize a 3d numoy array simiiar to tf.image.per_image_standardization"""
    mean = image.mean()
    stddev = image.std()
    adjusted_stddev = max(stddev, 1.0/math.sqrt(image.size))
    standardized_image = (image - mean) / adjusted_stddev
    
    return standardized_image
    
def resize(img, height, width):
    # resize image
    dim = (width, height)    
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
    print("Original shape is {}".format(img.shape))
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
        
    #Give an id field
    boxes["box_id"] = np.arange(boxes.shape[0])
    
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

def create_boxes(plot_data, size=1):
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
    
    fixed_boxes["box_id"] = fixed_boxes.index.to_series().apply(lambda x: "fixed_box_{}".format(x))
    
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
    
    ##If no remaining boxes just take a box around center
    missing_ids = plot_data[~plot_data.individual.isin(merged_boxes.individual)]
    
    if not missing_ids.empty:
        created_boxes= create_boxes(missing_ids)
        merged_boxes = merged_boxes.append(created_boxes)
    
    #If there are multiple boxes per point, take the center box
    grouped = merged_boxes.groupby("individual")
    
    cleaned_boxes = []
    for value, group in grouped:
        choosen_box = choose_box(group, plot_data)
        cleaned_boxes.append(choosen_box)
    
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_boxes),crs=merged_boxes.crs)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
    ##if there are multiple points per box, take the tallest point.
    cleaned_points = []
    for value, group in merged_boxes.groupby("box_id"):
        if group.shape[0] > 1:
            print("removing {} points for within a deepforest box".format(group.shape[0]-1))
            cleaned_points.append(group[group.height == group.height.max()])
        else:
            cleaned_points.append(group)
     
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_points),crs=merged_boxes.crs)
        
    #assert plot_data.shape[0] == merged_boxes.shape[0]
    return merged_boxes

def run(plot, df, savedir, rgb_pool=None, saved_model=None, deepforest_model=None):
    """wrapper function for dask, see main.py"""
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
    
    #Write merged boxes to file as an interim piece of data to inspect.
    predicted_trees.to_file("{}/{}_boxes.shp".format(savedir, plot))

def main(
    field_data,
    rgb_dir, 
    savedir,
    saved_model=None, 
    client=None, 
    shuffle=True):
    """Prepare NEON field data into tfrecords
    Args:
        field_data: shp file with location and class of each field collected point
        height: height in meters of the resized training image
        width: width in meters of the resized training image
        savedir: direcory to save predicted bounding boxes
        client: dask client object to use
    Returns:
        None: .shp bounding boxes are written to savedir
    """ 
    df = gpd.read_file(field_data)
    plot_names = df.plotID.unique()
    
    rgb_pool = glob.glob(rgb_dir, recursive=True)
    
    if client is not None:
        futures = []
        for plot in plot_names:
            future = client.submit(
                run,
                plot=plot,
                df=df,
                rgb_pool=rgb_pool,
                saved_model=saved_model,
                savedir=savedir
            )
            futures.append(future)
        
        wait(futures)
        
    else:
        from deepforest import deepforest        
        deepforest_model = deepforest.deepforest()
        deepforest_model.use_release()        
        for plot in plot_names:
            try:
                run(
                    plot=plot,
                    df=df,
                    rgb_pool=rgb_pool,  
                    saved_model=saved_model,
                    deepforest_model=deepforest_model
                )
            except Exception as e:
                print("Plot failed with {}".format(e))      
                traceback.print_exc()  
                continue
    
if __name__ == "__main__":
    #Generate the training data shapefiles
    ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    lookup_glob = "/orange/ewhite/NeonData/**/CanopyHeightModelGtif/*.tif"
    
    #Read config from top level dir
    config = parse_yaml("{}/conf/tree_config.yml".format(ROOT))
    
    #create dask client
    client = start_cluster.start(cpus=config["cpu_workers"], mem_size="10GB")
    #client = None
    
    #Create train test split
    create_training_shp.train_test_split(ROOT, lookup_glob, n=config["train"]["resampled_per_taxa"], client=client, regenerate=False)

    #test data
    main(
        field_data=config["evaluation"]["ground_truth_path"],    
        rgb_dir=config["rgb_sensor_pool"],
        client=client,
        savedir="{}/data/deepforest_boxes/test/".format(ROOT),
        saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    )
    
    print("Evaluation records complete")
    #clean out client of any objects
    client.restart()
    
    ##train data
    main(
        field_data=config["train"]["ground_truth_path"],      
        rgb_dir=config["rgb_sensor_pool"],
        client=client,
        savedir="{}/data/deepforest_boxes/train/".format(ROOT),        
        saved_model="/home/b.weinstein/miniconda3/envs/DeepTreeAttention_DeepForest/lib/python3.7/site-packages/deepforest/data/NEON.h5"
    )
    
