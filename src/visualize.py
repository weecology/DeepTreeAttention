#visualize
from descartes import PolygonPatch
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os
import pandas as pd
import rasterio
from rasterio.plot import show
from src import neon_paths
import tempfile

def index_to_example(index, test, test_crowns, test_points, rgb_pool, comet_experiment):
    """Function to plot an RGB image, the NEON field point and the deepforest crown given a test index
    Args:
        index: pandas index .loc for test.csv
        test_csv (pandas df): dataframe from data.py
        test_crowns (geopandas gdf): see generate.py
        test_points (pandas df): see generate.py
        rgb_pool: config glob path to search for rgb images, see config.yml
        experiment: comet_experiment
    Returns:
        image_name: name of file
        sample_id: comet id
    """
    tmpdir = tempfile.gettempdir()
    individual = test.loc[index]["individual"]
    
    fig = plt.figure(0)
    ax = fig.add_subplot(1, 1, 1)                
    geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
    left, bottom, right, top = geom.bounds
    
    #Find image
    img_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=geom.bounds)
    src = rasterio.open(img_path)
    img = src.read(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
    img_transform = src.window_transform(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
    
    #Plot crown
    patches = [PolygonPatch(geom, edgecolor='red', facecolor='none')]
    show(img, ax=ax, transform=img_transform)                
    ax.add_collection(PatchCollection(patches, match_original=True))
    
    #Plot field coordinate
    stem = test_points[test_points.individual == individual]
    stem.plot(ax=ax)
    
    image_name = "{}/{}_confusion.png".format(tmpdir,individual)
    plt.title("{}".format(individual))
    
    plt.savefig(image_name)
    results = comet_experiment.log_image(image_name, name = "{}".format(individual))
    src.close()
    plt.close("all")
    
    # Return sample, assetId (index is added automatically)
    return {"sample": image_name, "assetId": results["imageId"]}

def confusion_matrix(comet_experiment, results, species_label_dict, test, test_points, test_crowns, rgb_pool, name):
    #Confusion matrix
    comet_experiment.log_confusion_matrix(
        name=name,
        results.label.values,
        results.pred_label_top1.values,
        labels=list(species_label_dict.keys()),
        max_categories=len(species_label_dict.keys()),
        index_to_example_function=index_to_example,
        test=test,
        test_points=test_points,
        test_crowns=test_crowns,
        rgb_pool=rgb_pool,
        comet_experiment=comet_experiment)
    