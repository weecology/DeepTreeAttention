"""Visualization tools"""
#From https://gist.github.com/jakevdp/91077b0cae40f8f8244a
import matplotlib.pyplot as plt
import numpy as np
from skimage import exposure
import pandas as pd

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

def site_confusion(y_true, y_pred, sites):
    """What proportion of misidentified species come from the same site?"""
    
    #Create site x species matrix in the test data
    df = pd.DataFrame({"site":sites,"species":y_true})
    site_lists = df.groupby("species").site.unique().to_dict()
    
    within_site = 0
    cross_site = 0    
    for index, value in enumerate(y_pred):

        #If not correctly predicted
        if not value == y_true[index]:
            correct_sites = site_lists[y_true[index]]
            
            try:
                incorrect_site = site_lists[y_pred[index]]
            except Exception as e:
                print(e)
                continue
        
            #Do they co-occur?
            site_overlap = any([site in incorrect_site for site in correct_sites])
            if site_overlap:
                within_site +=1
            else:
                cross_site +=1   
        else:
            pass
    
    #don't divide by zero
    if within_site + cross_site == 0:
        return 0
    
    #Get proportion of within site error
    proportion_within = within_site/(within_site + cross_site)
    
    return proportion_within
