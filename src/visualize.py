import glob
import rasterio
import tempfile
import matplotlib.pyplot as plt

from rasterio.plot import show
from geopandas import GeoSeries

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
    
def index_to_example(index, test, test_crowns, test_points, comet_experiment, crop_dir):
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
    tmpdir = tempfile.gettempdir()
    individual = test.loc[index]["individual"]
    point = test_points[test_points.individual == individual].geometry.iloc[0]
    geom = test_crowns[test_crowns.individual == individual].geometry.iloc[0]
    img_path = test.loc[index]["RGB_tile"]
    crown_plot(img_path, geom, point)
    image_name = "{}/{}_confusion.png".format(tmpdir,individual)
    plt.title("{}".format(individual))
    
    plt.savefig(image_name)
    results = comet_experiment.log_image(image_name, name = "{}".format(individual))
    plt.close("all")
    
    # Return sample, assetId (index is added automatically)
    return {"sample": image_name, "assetId": results["imageId"]}

def confusion_matrix(comet_experiment, yhats, y, labels, test, test_points, test_crowns, name, crop_dir):
    #Confusion matrix
    comet_experiment.log_confusion_matrix(
        y_predicted=yhats,
        y_true=y,
        labels=labels,
        max_categories=len(labels),
        index_to_example_function=index_to_example,
        test=test,
        test_points=test_points,
        test_crowns=test_crowns,
        comet_experiment=comet_experiment,
        title=name,
        file_name="{}.json".format(name),
        crop_dir=crop_dir
    )

