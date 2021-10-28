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
from sklearn.manifold import TSNE

def index_to_example(index, test_csv, test_crowns, test_points, rgb_pool, comet_experiment):
    """Function to plot an RGB image, the NEON field point and the deepforest crown given a test index
    Args:
        index: pandas index .loc for test.csv
        test_csv (str): path to test.csv
        test_crowns (str): path to test_crowns.shp, see generate.py
        test_points (str): path to test_points.csv see generate.py
        rgb_pool: config glob path to search for rgb images, see config.yml
        experiment: comet_experiment
    Returns:
        image_name: name of file
        sample_id: comet id
    """
    tmpdir = tempfile.gettempdir()
    test = pd.read_csv(test_csv)
    test_crowns = gpd.read_file(test_crowns)
    test_points = gpd.read_file(test_points)
    individual = os.path.splitext(os.path.basename(test.loc[index]["image_path"]))[0]
    
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
    plt.savefig(image_name)
    results = comet_experiment.log_image(image_name, name = "{}".format(individual))
    src.close()
    plt.close("all")
    
    # Return sample, assetId (index is added automatically)
    return {"sample": image_name, "assetId": results["imageId"]}

def confusion_matrix(comet_experiment, results, species_label_dict, test_csv, test_points, test_crowns, rgb_pool):
    #Confusion matrix
    #comet_experiment.log_confusion_matrix(
        #results.label.values,
        #results.pred_label.values,
        #labels=list(species_label_dict.keys()),
        #max_categories=len(species_label_dict.keys()),
        #index_to_example_function=index_to_example,
        #test_csv=test_csv,
        #test_points=test_points,
        #test_crowns=test_crowns,
        #rgb_pool=rgb_pool,
        #comet_experiment=comet_experiment)

    comet_experiment.log_confusion_matrix(
        results.label.values,
        results.pred_label.values,
        labels=list(species_label_dict.keys()),
        max_categories=len(species_label_dict.keys()))

def n_colors(n):
    colors = []
    for x in range(n):
        color = list(np.random.choice(range(256), size=3)/255)
        colors.append(color)
    return colors 
    
def plot_2d_layer(features, labels=None, use_tsne=False):
    """Given a 2D tensor array and a list of labels, plot and optionally color
    Args:
        features: input feature matrix
        labels: label for each feature row
        use_tsne: Whether to first reduce dimensionality using tsne"""
    
    num_categories = max(np.unique(labels)) + 1   
    colors = n_colors(n = num_categories)
    
    if use_tsne:
        tsne = TSNE(2, verbose=1)
        tsne_proj = tsne.fit_transform(features)     
        fig, ax = plt.subplots(figsize=(8,8))
        for lab in range(num_categories):
            indices = labels==lab
            ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=colors[lab], label = lab ,alpha=0.5)
        ax.legend(fontsize='large', markerscale=2)
    else: 
        features = pd.DataFrame(features, columns=["a","b"])
        features["label"] = labels
        features["color"] = features.label.apply(lambda x: colors[x])
        
        fig = features.plot.scatter(x="a",y="b",color=features.color, alpha=0.75)
    
    return fig