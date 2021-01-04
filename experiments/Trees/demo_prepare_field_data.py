
#Testing script that lives outside of pytest due to DeepForest dependency in a new conda env. PYTHONPATH manually added to root dir for relative paths.
import os
import pytest
import tensorflow
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import pandas as pd
import glob

import prepare_field_data
from DeepTreeAttention.generators import boxes
from deepforest import deepforest

data_dir = os.path.dirname(prepare_field_data.__file__)
data_path = "{}/test_data/sample.shp".format(data_dir)
rgb_dir = "{}/test_data/rgb/*.tif".format(data_dir)

ROOT = os.path.dirname(os.path.dirname(data_dir))

rgb_pool = glob.glob(rgb_dir, recursive=True)

height = 100
width = 100

def test_empty_plot():
    #DeepForest prediction
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    plot_data = gpd.read_file(data_path)    
    rgb_sensor_path = prepare_field_data.find_sensor_path(bounds=plot_data.total_bounds, lookup_pool=rgb_pool)
    boxes = prepare_field_data.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_sensor_path, bounds=plot_data.total_bounds)

    #fake offset boxes by adding a scalar to the geometry
    boxes["geometry"] = boxes["geometry"].translate(100000)
        
    #Merge results with field data, buffer on edge 
    merged_boxes = gpd.sjoin(boxes, plot_data)
    
    #If no remaining boxes just take a box around center
    if merged_boxes.empty:
        merged_boxes= prepare_field_data.create_boxes(plot_data)
        
    #If there are multiple boxes, take the center box
    grouped = merged_boxes.groupby("individual")
    
    cleaned_boxes = []
    for value, group in grouped:
        choosen_box = prepare_field_data.choose_box(group, plot_data)
        cleaned_boxes.append(choosen_box)
    
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_boxes),crs=merged_boxes.crs)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
def test_process_plot():
    df = gpd.read_file(data_path)
    
    deepforest_model = deepforest.deepforest()
    deepforest_model.use_release()
    
    merged_boxes = prepare_field_data.process_plot(plot_data=df, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
    assert df.shape[0] >= merged_boxes.shape[0]
    
def test_run():
    df = gpd.read_file(data_path)
    plot = df.plotID.unique()[0]
    prepare_field_data.run(
        plot=plot,
        df = df,
        rgb_pool=rgb_pool,
    ) 
    
#Run tests
test_empty_plot()
test_process_plot()
test_run()
