#Test generate
from src import generate
import glob
import geopandas as gpd
import pandas as pd
from deepforest import main
import rasterio
import pytest
import os
import distributed
import numpy as np

def test_predict_trees(rgb_path, plot_data):
    m = main.deepforest()
    m.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=m, rgb_path=rgb_path, bounds=plot_data.total_bounds)
    assert not boxes.empty 

def test_empty_plot(rgb_path, plot_data):
    #DeepForest prediction
    deepforest_model = main.deepforest()
    deepforest_model.use_release(check_release=False)
    boxes = generate.predict_trees(deepforest_model=deepforest_model, rgb_path=rgb_path, bounds=plot_data.total_bounds)

    #fake offset boxes by adding a scalar to the geometry
    boxes["geometry"] = boxes["geometry"].translate(100000)
        
    #Merge results with field data, buffer on edge 
    merged_boxes = gpd.sjoin(boxes, plot_data)
    
    assert merged_boxes.empty

    #If no remaining boxes just take a box around center
    merged_boxes= generate.create_boxes(plot_data)
        
    #If there are multiple boxes, take the center box
    grouped = merged_boxes.groupby("individual")
    
    cleaned_boxes = []
    for value, group in grouped:
        choosen_box = generate.choose_box(group, plot_data)
        cleaned_boxes.append(choosen_box)
    
    merged_boxes = gpd.GeoDataFrame(pd.concat(cleaned_boxes),crs=merged_boxes.crs)
    merged_boxes = merged_boxes.drop(columns=["xmin","xmax","ymin","ymax"])
    
    assert not merged_boxes.empty
    
def test_process_plot(rgb_pool, sample_crowns):
    df = gpd.read_file(sample_crowns)
    deepforest_model = main.deepforest()
    deepforest_model.use_release(check_release=False)
    
    merged_boxes, boxes = generate.process_plot(plot_data=df, rgb_pool=rgb_pool, deepforest_model=deepforest_model)
    assert df.shape[0] >= merged_boxes.shape[0]
    assert len(merged_boxes.box_id.unique()) == merged_boxes.shape[0]
    
def test_run(tmpdir, sample_crowns, rgb_pool):
    df = gpd.read_file(sample_crowns)
    df["geometry"] = df.centroid
    df = df.drop(columns="box_id")
    plot = df.plotID.unique()[0]
    generate.run(
        plot=plot,
        df = df,
        rgb_pool=rgb_pool,
        savedir=tmpdir,
        raw_box_savedir=tmpdir
    ) 
    
    assert len(glob.glob("{}/*.shp".format(tmpdir))) > 0    

@pytest.mark.parametrize("use_client",[True,False])
def test_generate_crops(tmpdir, ROOT, config, rgb_path, rgb_pool, sample_crowns, use_client):
    if use_client:
        client = distributed.Client()
    else:
        client=None

    gdf = gpd.read_file(sample_crowns)
    gdf.geometry = gdf.geometry.buffer(1)
    gdf["RGB_tile"] = rgb_path
    gdf["box_id"] = gdf.index
    img_pool = glob.glob("{}*.tif".format(config["HSI_tif_dir"]))

    annotations = generate.generate_crops(
        gdf=gdf,
        rgb_pool=rgb_pool,
        convert_h5=False,
        img_pool=img_pool,
        h5_pool=None,
        suffix="HSI",
        client=client,
        savedir=tmpdir)
    
    assert annotations.tile_year.unique() == "2019" 
