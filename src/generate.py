#Convert NEON field sample points into bounding boxes of cropped image data for model training
import glob
import geopandas as gpd
import rasterio
import numpy as np
import shapely
import pandas as pd
from src.neon_paths import find_sensor_path
from src import start_cluster
from src import patches
from distributed import wait   
from deepforest import main    
import cv2

def predict_trees(deepforest_model, rgb_path, bounds, expand=40):
    """Predict an rgb path at specific utm bounds
    Args:
        deepforest_model: a deepforest model object used for prediction
        rgb_path: full path to image
        bounds: utm extent given by geopandas.total_bounds
        expand: numeric meters to add to edges to reduce edge effects
        """
    #DeepForest is trained on 400m crops, easiest to mantain this approximate size centered on points
    left, bottom, right, top = bounds
    expand_width = (expand - (right - left))/2
    left = left - expand_width
    right = right + expand_width
    
    expand_height = (expand - (top - bottom))/2 
    bottom = bottom - expand_height
    top = top + expand_height 
    
    src = rasterio.open(rgb_path)
    pixelSizeX, pixelSizeY  = src.res    
    img = src.read(window=rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform))
    src.close()
    
    #roll to channels last
    img = np.rollaxis(img, 0,3)
    boxes = deepforest_model.predict_image(image = img, return_plot=False)
    
    if boxes is None:
        return boxes

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
    
def process_plot(plot_data, rgb_pool, deepforest_model=None):
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

    if boxes is None:
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
        
    return merged_boxes, boxes 

def run(plot, df, savedir, raw_box_savedir, rgb_pool=None, saved_model=None, deepforest_model=None):
    """wrapper function for dask, see main.py"""
    
    if deepforest_model is None:
        from deepforest import main
        deepforest_model = main.deepforest()
        try:
            deepforest_model.use_release()
        except:
            deepforest_model.use_release(check_release=False)        

    #Filter data and process
    plot_data = df[df.plotID == plot]
    try:
        predicted_trees, raw_boxes = process_plot(plot_data, rgb_pool, deepforest_model)
    except ValueError as e:
        print(e)
        return None
    
    if predicted_trees.empty:
        return None
    
    #Write merged boxes to file as an interim piece of data to inspect.
    if savedir is not None:
        predicted_trees.to_file("{}/{}_boxes.shp".format(savedir, plot))
    
    if raw_box_savedir is not None:
        raw_boxes.to_file("{}/{}_boxes.shp".format(raw_box_savedir, plot))
    
    return predicted_trees

def points_to_crowns(
    field_data,
    rgb_dir, 
    savedir,
    raw_box_savedir,
    client=None):
    """Prepare NEON field data int
    Args:
        field_data: shp file with location and class of each field collected point
        rgb_dir: glob to search RGB images
        savedir: direcory to save predicted bounding boxes
        raw_box_savedir: directory save all bounding boxes in the image
        client: dask client object to use
    Returns:
        None: .shp bounding boxes are written to savedir
    """ 
    df = gpd.read_file(field_data)
    plot_names = df.plotID.unique()
    
    rgb_pool = glob.glob(rgb_dir, recursive=True)
    results = []    
    if client:
        futures = []
        for plot in plot_names:
            future = client.submit(
                run,
                plot=plot,
                df=df,
                rgb_pool=rgb_pool,
                savedir=savedir,
                raw_box_savedir=raw_box_savedir
            )
            futures.append(future)
            
        wait(futures)
        
        for x in futures:
            try:
                x.result()
                results.append()
            except:
                continue
    else:
        #IMPORTS at runtime due to dask pickling, kinda ugly.
        deepforest_model = main.deepforest()  
        deepforest_model.use_release(check_release=False)
        
        for plot in plot_names:
            try:
                result = run(plot=plot, df=df, savedir=savedir, raw_box_savedir=raw_box_savedir, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
                results.append(result)
            except Exception as e:
                print("{} failed with {}".format(plot, e))
    results = pd.concat(results)
    
    return results

def write_crop(row, img_path, savedir, label_dict, size):
    """Wrapper to write a crop"""
    counter = 0
    crops = patches.crown_to_pixel(crown=row["geometry"], img_path=img_path, width=size, height=size)
    filenames = []
    labels = []
    for x in crops:
        label = label_dict[row["taxonID"]]
        labels.append(label)
        filename = "{}/{}_{}.png".format(savedir,row["individual"], counter)
        channnels_last = np.rollaxis(x,0,3)
        cv2.imwrite(filename, channnels_last)
        filenames.append(filename)
        counter = counter + 1
    annotation = pd.DataFrame({"image_path":filenames, "label":labels})
    
    return annotation

def generate_crops(gdf, sensor_glob, savedir, label_dict, size, client=None):
    """
    Given a shapefile of crowns in a plot, create pixel crops and a dataframe of unique names and labels"
    Args:
        shapefile: a .shp with geometry objects and an taxonID column
        savedir: path to save image crops
        img_pool: glob to search remote sensing files. This can be either RGB of .tif hyperspectral data, as long as it can be read by rasterio
        label_dict (dict): taxonID -> numeric order
        size: number of pixel width and height for the windows
        client: optional dask client
    Returns:
       annotations: pandas dataframe of filenames and individual IDs to link with data
    """
    annotations = []
    
    img_pool = glob.glob(sensor_glob, recursive=True)
    
    if client:
        for index, row in gdf.iterrows():
            try:
                img_path = find_sensor_path(lookup_pool = img_pool, bounds = row.geometry.bounds)            
            except:
                print("Cannot find matching file in image pool for {}".format(row.head()))      
                continue
            
            futures = client.submit(write_crop,row=row,img_path=img_path, label_dict=label_dict, size=size, savedir=savedir)
        
        wait(futures)
        for x in futures:
            annotation = x.result()
            annotations.append(annotation)
    else:
        for index, row in gdf.iterrows():
            try:
                img_path = find_sensor_path(lookup_pool = img_pool, bounds = row.geometry.bounds)            
            except:
                print("Cannot find matching file in image pool for {}".format(row.head()))      
                continue      
            annotation = write_crop(row=row, img_path=img_path, savedir=savedir, label_dict=label_dict, size=size)
            annotations.append(annotation)
    
    annotations = pd.concat(annotations)
        
    return annotations
        