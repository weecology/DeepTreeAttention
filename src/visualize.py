import glob
import rasterio
import tempfile
import matplotlib.pyplot as plt

from rasterio.plot import show
from geopandas import GeoSeries

from src import neon_paths
from src.utils import load_image

def crown_plot(img_path, geom, point):
    #Find image
    fig, ax = plt.subplots(figsize=(4, 4))
    src = rasterio.open(img_path)
    
    #Plot crown
    show(src, ax=ax)                    
    g = GeoSeries([geom])
    g.plot(ax=ax, facecolor="none", edgecolor="red")
    
    #Plot field coordinate
    p = GeoSeries([point])
    p.plot(ax=ax)
    
    
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
    point = test_points[test_points.individual == individual]
    geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
    img_path = test.loc[index]["RGB_image_path"]
    figure = crown_plot(image_path, geom, point)

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
        results.label.values,
        results.ens_label.values,
        labels=list(species_label_dict.keys()),
        max_categories=len(species_label_dict.keys()),
        index_to_example_function=index_to_example,
        test=test,
        test_points=test_points,
        test_crowns=test_crowns,
        rgb_pool=rgb_pool,
        comet_experiment=comet_experiment,
        title=name
    )
    