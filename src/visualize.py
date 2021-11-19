#visualize
from descartes import PolygonPatch
import glob
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
from sklearn.decomposition import PCA

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

def n_colors(n, set_color_seed=True):
    colors = []
    if set_color_seed:
        np.random.seed(2)
    for x in range(n):
        color = list(np.random.choice(range(256), size=3)/255)
        colors.append(color)
    return colors 
    
def plot_2d_layer(features, labels=None, use_pca=False, set_color_seed=True, size_weights=[]):
    """Given a 2D tensor array and a list of labels, plot and optionally color
    Args:
        features: input feature matrix
        labels: numeric label for each feature row
        use_pca: Whether to first reduce dimensionality using pca
        size_weights: array the same length as the features that gives the size multiplier for each point
        """
    num_categories = max(np.unique(labels)) + 1   
    colors = n_colors(n = num_categories, set_color_seed=set_color_seed)
    
    if len(size_weights) > 0:
        s = 7.0 * size_weights + 1
    else:
        s = 4
    if use_pca:
        pca = PCA(2)
        #flatten features
        features = np.reshape(features, (features.shape[0], features.shape[1] * features.shape[2] *features.shape[3]))
        pca_proj = pca.fit_transform(features)     
        fig, ax = plt.subplots(figsize=(8,8))
        for index, lab in enumerate(np.unique(labels)):
            indices = labels==lab
            if len(size_weights) > 0:
                point_size = s[indices]
            else:
                point_size = s
            ax.scatter(pca_proj[indices,0],pca_proj[indices,1], c=colors[index], label = lab ,alpha=0.7, s=point_size)
        ax.legend(fontsize='large', markerscale=2)
    else: 
        fig, ax = plt.subplots()                
        for index, label in enumerate(np.unique(labels)):
            x = features[labels == label, 0]
            y = features[labels == label, 1]
            
            if len(size_weights) > 0:
                point_size = s[labels == label]
            else:
                point_size = s
                
            ax.scatter(x,y,color=colors[index], alpha=0.7, s=point_size, label=label)   
            ax.legend(markerscale=2)

def plot_points_and_crowns(df, ROOT, plot_n_individuals, config, experiment):
    """Plot a dataframe of results"""
    #load image pool and crown predicrions
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)
    test_points = gpd.read_file("{}/data/processed/canopy_points.shp".format(ROOT))   
    test_crowns = gpd.read_file("{}/data/processed/crowns.shp".format(ROOT))   
    tmpdir = tempfile.gettempdir()
    
    plt.ion()
    for index, row in df.head(n=plot_n_individuals).iterrows():
        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)                
        individual = row["individual"]
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
        
        plt.savefig("{}/{}.png".format(tmpdir, row["individual"]))
        experiment.log_image("{}/{}.png".format(tmpdir, row["individual"]), name = "crown: {}, True: {}, Predicted {}".format(row["individual"], row.true_taxa,row.pred_taxa))
        src.close()
        plt.close("all")
    plt.ioff()    