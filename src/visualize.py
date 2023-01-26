from descartes import PolygonPatch
import glob
import rasterio
import tempfile
import matplotlib.pyplot as plt

from matplotlib.collections import PatchCollection
from rasterio.plot import show
import torch
import pandas as pd
from src import neon_paths

def plot_examples(df, test_crowns, plot_n_individuals, test_points, rgb_sensor_pool, experiment):
    """Create a visualization of crowns and points overlaid on RGB"""
    tmpdir = tempfile.gettempdir()
    #load image pool and crown predicrions
    rgb_pool = glob.glob(rgb_sensor_pool, recursive=True)  
    plt.ion()
    for index, row in df.sample(n=plot_n_individuals).iterrows():
        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)                
        individual = row["individual"]
        try:
            geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
        except Exception as e:
            print("Cannot find individual {} in crowns.shp with schema {}".format(individual, test_crowns.head()))
            break
            
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
        experiment.log_image("{}/{}.png".format(tmpdir, row["individual"]), name = "crown: {}, True: {}, Predicted {}".format(row["individual"], row.true_taxa,row.pred_taxa_top1))
        src.close()
        plt.close("all")
    plt.ioff()
            
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

def visualize_consistency(prob_strong, prob_weak):
    """Create a barplot of class labels"""
    prob_array = torch.stack([prob_strong.squeeze(),prob_weak.squeeze()]).T
    prob_array = pd.DataFrame(prob_array.detach().cpu().numpy())
    prob_array.columns = ["weak","strong"]
    return prob_array.plot(kind="bar")
