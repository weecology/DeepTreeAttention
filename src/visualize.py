#visualize
from descartes import PolygonPatch
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import os
import rasterio
from rasterio.plot import show
from src import neon_paths
from src import utils
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

def confusion_matrix(comet_experiment, results, species_label_dict, test, test_points, test_crowns, rgb_pool):
    #Confusion matrix
    comet_experiment.log_confusion_matrix(
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
    
def rgb_plots(df, config, test_crowns, test_points, plot_n_individuals=1, experiment=None):
    """Create visualization of predicted crowns and label
    Args:
        df: a dataframe returned from main.predict_dataloader
    Returns:
        None: plots are generated and uploaded to experiment
    """
    #load image pool and crown predictions
    tmpdir = tempfile.gettempdir()
    rgb_pool = glob.glob(config["rgb_sensor_pool"], recursive=True)            
    plt.ion()
    if plot_n_individuals > df.shape[0]:
        plot_n_individuals = df.shape[0]
        
    for index, row in df.sample(n=plot_n_individuals).iterrows():    
        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)                
        individual = row["individual"]
        try:
            geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
        except:
            raise ValueError("Cannot find individual {} in test crowns, example format: {}".format(individual,test_crowns.head().individual))
        
        left, bottom, right, top = geom.bounds
        
        #Find RGB image
        img_path = neon_paths.find_sensor_path(lookup_pool=rgb_pool, bounds=geom.bounds)
        src = rasterio.open(img_path)
        img = src.read(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
        img_transform = src.window_transform(window=rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform))  
        
        #Plot crown
        patches = [PolygonPatch(geom, edgecolor='red', facecolor='none')]
        show(img, ax=ax, transform=img_transform)                
        ax.add_collection(PatchCollection(patches, match_original=True))
        
        #Plot field coordinate
        stem = test_points[test_points.individualID == individual]
        stem.plot(ax=ax)
        
        if experiment:
            plt.savefig("{}/{}.png".format(tmpdir, row["individual"]))
            experiment.log_image("{}/{}.png".format(tmpdir, row["individual"]), name="crown: {}, True: {}, Predicted {}".format(row["individual"], row.true_taxa, row.pred_taxa_top1))
        src.close()
        plt.close("all")
    plt.ioff()    

def plot_spectra(df, crop_dir, plot_n_individuals=20, experiment=None):
    """Create pixel spectra figures from a results object
    Args:
       df: pandas dataframe generated by main.predict_dataloader
    """
    tmpdir = tempfile.gettempdir()    
    if plot_n_individuals > df.shape[0]:
        plot_n_individuals = df.shape[0]
    for index, row in df.sample(n=plot_n_individuals).iterrows():
        #Plot spectra
        HSI_path = os.path.join(crop_dir,"{}.tif".format(row["individual"]))
        hsi_sample = utils.load_image(img_path=HSI_path, image_size=11)
        for x in hsi_sample.reshape(hsi_sample.shape[0], np.prod(hsi_sample.shape[1:])).T:
            plt.plot(x)
        if experiment:
            plt.savefig("{}/{}_spectra.png".format(tmpdir, row["individual"]))            
            experiment.log_image("{}/{}_spectra.png".format(tmpdir, row["individual"]), name="{}, {} Predicted {}".format(row["individual"], row.true_taxa, row.pred_taxa_top1))
        plt.close()    