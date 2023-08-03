import glob
import rasterio
import tempfile
import numpy as np
import matplotlib.pyplot as plt

from rasterio.plot import show
from geopandas import GeoSeries

def view_plot(crowns, unfiltered_points, CHM_points, image_path, savedir):
    """Visualize the bounding box crowns and filtered points for each plotID"""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('equal')
    left, bottom, right, top = crowns.total_bounds
    src = rasterio.open(image_path)
    window = rasterio.windows.from_bounds(left-10, bottom-10, right+10, top+10, transform=src.transform)
    img = src.read(window=window) 
    win_transform = src.window_transform(window)
    show(img, ax=ax, transform=win_transform)   

    #Plot crown
    ax = crowns.plot(ax=ax, facecolor="none", edgecolor="red")
    
    #Plot field coordinate
    ax = unfiltered_points.plot(ax=ax, color="red",markersize=80)
    ax = CHM_points.plot(ax=ax, color="black")
    plt.title(crowns.plotID.unique()[0])

    for x, y, label in zip(CHM_points.geometry.x, CHM_points.geometry.y, CHM_points.individual):
        ax.annotate(label, xy=(x, y), xytext=(3, 3), textcoords="offset points")


    plt.savefig("{}/{}.png".format(savedir, crowns.plotID.unique()[0]))

def crown_plot(img_path, geom, point, expand=3):
    #Find image
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.axis('equal')
    left, bottom, right, top = geom.buffer(expand).bounds 
    src = rasterio.open(img_path)
    window = rasterio.windows.from_bounds(left, bottom, right, top, transform=src.transform)
    img = src.read(window=window) 
    win_transform = src.window_transform(window)
    show(img, ax=ax, transform=win_transform)                    

    #Plot crown
    g = GeoSeries([geom])
    ax = g.plot(ax=ax, facecolor="none", edgecolor="red")
    
    #Plot field coordinate
    p = GeoSeries([point])
    p.plot(ax=ax)
    
def index_to_example(index, individuals,RGB_tiles, test_crowns, test_points, comet_experiment):
    """Function to plot an RGB image, the NEON field point and the deepforest crown given a test index
    Args:
        index: pandas index .loc for test.csv
        test_csv (pandas df): dataframe from data.py
        test_crowns (geopandas gdf): see generate.py
        test_points (pandas df): see generate.py
        experiment: comet_experiment
    Returns:
        image_name: name of file
        sample_id: comet id
    """
    try:
        tmpdir = tempfile.gettempdir()
        individual = individuals[index]
        point = test_points[test_points.individual == individual].geometry.iloc[0]
        geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
        img_path =  RGB_tiles[index]
        crown_plot(img_path, geom, point)
        image_name = "{}/{}_confusion.png".format(tmpdir,individual)
        plt.title("{}".format(individual))
        
        plt.savefig(image_name)
        results = comet_experiment.log_image(image_name, name = "{}".format(individual))
        plt.close("all")
        # Return sample, assetId (index is added automatically)
        return {"sample": image_name, "assetId": results["imageId"]}
    except:
        return {"sample": None, "assetId": None}
        
def view_samples(ds, samples, comet_logger=None):
    """Visualize a few samples as images and spectra"""
    tmpdir = tempfile.gettempdir()
    for x in range(10):
        individual, inputs, label = ds[x]
        three_band_HSI = inputs["HSI"][[55,90,115], :,:]
        for x in inputs["HSI"].reshape(inputs["HSI"].shape[0], np.prod(inputs["HSI"].shape[1:])).T:
                plt.plot(x)
        if comet_logger:
                comet_logger.experiment.log_image(
                image_data=three_band_HSI.numpy(),
                    name=individual,
                    image_channels="first"
                    )
                plt.savefig("{}/{}_spectra.png".format(tmpdir, individual))
                comet_logger.experiment.log_image("{}/{}_spectra.png".format(tmpdir, individual))

def confusion_matrix(comet_experiment, yhats, y, labels, individuals,RGB_tiles, test_points, test_crowns, name):
    #Confusion matrix
    comet_experiment.log_confusion_matrix(
        y_predicted=yhats,
        y_true=y,
        individuals=individuals,
        RGB_tiles=RGB_tiles,
        labels=labels,
        max_categories=len(labels),
        index_to_example_function=index_to_example,
        test_points=test_points,
        test_crowns=test_crowns,
        comet_experiment=comet_experiment,
        title=name,
        file_name="{}.json".format(name),
    )

