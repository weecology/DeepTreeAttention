#test outliers
import os
from src import outlier
from src import visualize
from src.models import outlier_detection
from src import simulation
from src import data
import numpy as np
import tempfile
import pandas as pd

def test_distance_from_centroids():
    #Create a cluster with an outlier at a perfect 3,4,5 triangle away for easy calculation
    features = np.zeros([100,2])
    features = np.vstack([features, np.array([3,4])])
    labels = np.repeat(0, 101)
    centroids = {}
    centroids[0] = [0,0]
    cov = outlier.calculate_covariance(features, labels)
    distances = outlier.distance_from_centroids(features, centroids, labels, cov)
    true_distances = [0 for x in range(100)]
    true_distances.append(5)

    #view
    colors = [int(x > 2) for x in distances]
    visualize.plot_2d_layer(features=features, labels=colors)

def config():
    #Turn of CHM filtering for the moment
    config = data.read_config(config_path="{}/config.yml".format(ROOT))
    config["min_CHM_height"] = None
    config["iterations"] = 1
    config["rgb_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["HSI_sensor_pool"] = "{}/tests/data/*.tif".format(ROOT)
    config["min_samples"] = 1
    config["crop_dir"] = tempfile.gettempdir()
    config["bands"] = 3
    config["classes"] = 2
    config["top_k"] = 1
    config["convert_h5"] = False
    config["plot_n_individuals"] = 1
    
    return config

#Data module
def dm(config):
    csv_file = "{}/tests/data/sample_neon.csv".format(ROOT)           
    if not "GITHUB_ACTIONS" in os.environ:
        regen = False
    else:
        regen = True
    
    return dm

def test_predict_outliers():
    ROOT = os.path.dirname(os.path.dirname(simulation.__file__))
    config = data.read_config("{}/config.yml".format(ROOT))
    config["autoencoder_epochs"] = 1
    config["classifier_epochs"] = 1
    config["workers"] = 0
    config["gpus"] = 0
    config["fast_dev_run"] = False
    config["proportion_switch"] = 0.1
    config["samples"] = 100

    model = outlier_detection.autoencoder(bands=3, classes=2, config=config)
    annotations = pd.read_csv("{}/tests/data/processed/train.csv".format(ROOT))
    results = outlier.predict_outliers(model, annotations=annotations, config=config)
    assert annotations.shape[0] == results.shape[0] 
    assert "predicted_outlier" in results.columns