"""Visualization tools"""
import geopandas as pd
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import os
import pandas as pd
import geopandas as gpd

def normalize(array):
    """Normalizes numpy arrays into scale 0.0 - 1.0"""
    array_min, array_max = array.min(), array.max()
    return ((array - array_min) / (array_max - array_min))


def plot_prediction(image, label, prediction):
    """Plot an image with labels, optionally create a three band composite
    Args:
        image: a rgb or multiband image
        label: true class
        prediction: predicted class
        ls_pct: linear stretch of three band
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)

    #check if hyperspec and create three band false color.
    if image.shape[2] > 3:
        plot_image = image[:, :, [11, 55, 113]]
        for band in np.arange(plot_image.shape[2]):
            plot_image[:, :, band] = normalize(plot_image[:, :, band])
            plot_image.astype("float")
    else:
        plot_image = image.astype(int)

    ax.imshow(plot_image)
    ax.set_title("True: {}, Predicted: {} ".format(label, prediction))

    return fig


def create_raster(results):
    """Reshape a set of predictions from DeepTreeAttention.predict into a raster image"""
    #Create image
    rowIDs = results['row']
    colIDs = results['col']
    predicted_raster = np.zeros((rowIDs.max() + 1, colIDs.max() + 1))
    predicted_raster[rowIDs, colIDs] = results["label"]
    predicted_raster = predicted_raster.astype("uint16")

    return predicted_raster


def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

def plot_crown_position(path, y_true, y_pred, box_index):
    """Plot the errors by their crown position category
    Args:
        path: file path to the original data used to generate tfrecords
        model: keras model object to predict
        eval_dataset_with_index: tf dataset that yields data and index to match to original
        submodel: Logical, whether to replicate labels for submodel data (3x)
    Returns:
        matplotlib axes
    """
    train_shp = gpd.read_file(path)
    
    #get canopy dictionary
    canopy_dict = {}
    for index in box_index:
        canopy_dict[index] = train_shp[train_shp.index.astype(str) == index].canopyPosi.values[0]
    
    ax = canopyPosition_barplot(y_true, y_pred, box_index, canopy_dict)
    
    return ax
    
def canopyPosition_barplot(y_true, y_pred, box_index, canopydict):
    results = pd.DataFrame({"true":y_true,"predicted":y_pred, "box_index":box_index})
    
    results["canopyPosition"] = results.box_index.apply(lambda x: canopydict[x])
    results["match"] = (results["true"] == results["predicted"])
    
    summary = results.groupby(["canopyPosition","match"]).size().reset_index(name="count")
    ax = summary.pivot("canopyPosition","match","count").plot.bar()
    plt.xticks(rotation=-20)
    plt.xticks(fontsize=8)
    
    return ax    