#Convert NEON field sample points into bounding boxes of cropped image data for model training
import glob
import geopandas as gpd
import rasterio
import numpy as np
import shapely
import os
import pandas as pd
from src.neon_paths import find_sensor_path, lookup_and_convert, bounds_to_geoindex
from src import patches
from distributed import wait   
from deepforest import main    
import traceback
import warnings
warnings.filterwarnings('ignore')

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
    """Given a set of overlapping bounding boxes and predictions, just choose the closest to stem box by centroid if there are multiples"""
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
            print("removing {} points from {} within a deepforest box {}".format(group.shape[0]-1, group.plotID.unique(),group.box_id.unique()))
            selected_point = group[group.height == group.height.max()]
            if selected_point.shape[0] > 1:
                try:
                    selected_point = selected_point[selected_point.CHM_height == selected_point.CHM_height.max()]
                except:
                    selected_point.head(1)
            cleaned_points.append(selected_point)
        else:
            cleaned_points.append(group)
     
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_points),crs=merged_boxes.crs)
    
    #Add tile information
    boxes["RGB_tile"] = rgb_sensor_path
    merged_boxes["RGB_tile"] = rgb_sensor_path

    return merged_boxes, boxes 

def run(plot, df, savedir, raw_box_savedir, rgb_pool=None, saved_model=None, deepforest_model=None):
    """wrapper function for dask, see main.py"""
    
    if deepforest_model is None:
        from deepforest import main
        deepforest_model = main.deepforest()
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
                result = x.result()
                results.append(result)
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
    
    #In case any contrib data has the same CHM and height and sitting in the same deepforest box.Should be rare.
    results = results.groupby(["plotID","box_id"]).apply(lambda x: x.head(1)).reset_index(drop=True)
    
    return results

def write_crop(row, img_path, savedir, replace=True):
    """Wrapper to write a crop based on size and savedir"""
    if replace == False:
        filename = "{}/{}.tif".format(savedir, row["individual"])
        file_exists = os.path.exists(filename)
        if file_exists:
            annotation = pd.DataFrame({"image_path":[filename], "taxonID":[row["taxonID"]], "plotID":[row["plotID"]], "individualID":[row["individual"]], "siteID":[row["siteID"]]})            
            return annotation            
        else:
            filename = patches.crop(bounds=row["geometry"].bounds, sensor_path=img_path, savedir=savedir, basename=row["individual"])  
    else:
        filename = patches.crop(bounds=row["geometry"].bounds, sensor_path=img_path, savedir=savedir, basename=row["individual"])
        annotation = pd.DataFrame({"image_path":[filename], "taxonID":[row["taxonID"]], "plotID":[row["plotID"]], "individualID":[row["individual"]], "RGB_tile":[row["RGB_tile"]], "siteID":[row["siteID"]],"box_id":[row["box_id"]]})
        return annotation

def generate_crops(gdf, sensor_glob, savedir, rgb_glob, client=None, convert_h5=False, HSI_tif_dir=None, replace=True):
    """
    Given a shapefile of crowns in a plot, create pixel crops and a dataframe of unique names and labels"
    Args:
        shapefile: a .shp with geometry objects and an taxonID column
        savedir: path to save image crops
        img_pool: glob to search remote sensing files. This can be either RGB of .tif hyperspectral data, as long as it can be read by rasterio
        client: optional dask client
        convert_h5: If HSI data is passed, make sure .tif conversion is complete
        rgb_glob: glob to search images to match when converting h5s -> tif.
        HSI_tif_dir: if converting H5 -> tif, where to save .tif files. Only needed if convert_h5 is True
    Returns:
       annotations: pandas dataframe of filenames and individual IDs to link with data
    """
    annotations = []
    
    img_pool = glob.glob(sensor_glob, recursive=True)
    rgb_pool = glob.glob(rgb_glob, recursive=True)
    
    #There were erroneous point cloud .tif
    img_pool = [x for x in img_pool if not "point_cloud" in x]
    rgb_pool = [x for x in rgb_pool if not "point_cloud" in x]
     
    
    #Looking up the rgb -> HSI tile naming is expensive and repetitive. Create a dictionary first.
    gdf["geo_index"] = gdf.geometry.apply(lambda x: bounds_to_geoindex(x.bounds))
    tiles = gdf["geo_index"].unique()
    
    tile_to_path = {}
    for geo_index in tiles:
        try:
            #Check if h5 -> tif conversion is complete
            if convert_h5:
                if rgb_glob is None:
                    raise ValueError("rgb_glob is None, but convert_h5 is True, please supply glob to search for rgb images")
                else:
                    img_path = lookup_and_convert(rgb_pool=rgb_pool, hyperspectral_pool=img_pool, savedir=HSI_tif_dir,  geo_index = geo_index)
            else:
                img_path = find_sensor_path(lookup_pool = img_pool, geo_index = geo_index)  
        except:
            print("{} failed to find sensor path with traceback {}".format(geo_index, traceback.print_exc()))
            continue
        tile_to_path[geo_index] = img_path
            
    if client:
        futures = []
        for index, row in gdf.iterrows():
            try:
                img_path = tile_to_path[row["geo_index"]]
            except:
                continue
            future = client.submit(write_crop,row=row,img_path=img_path, savedir=savedir, replace=replace)
            futures.append(future)
            
        wait(futures)
        for x in futures:
            try:
                annotation = x.result()
                annotations.append(annotation)                
            except:
                print("Future failed with {}".format(traceback.print_exc()))
    else:
        for index, row in gdf.iterrows():
            try:
                img_path = tile_to_path[row["geo_index"]]
            except:
                continue
            try:
                annotation = write_crop(row=row, img_path=img_path, savedir=savedir, replace=replace)
            except Exception as e:
                print("index {} failed with {}".format(index,e))
                continue
    
            annotations.append(annotation)
    
    annotations = pd.concat(annotations)
        
    return annotations
        