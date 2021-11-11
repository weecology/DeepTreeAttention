#test outliers
from src import outlier
from src import visualize
import numpy as np

def test_distance_from_centroids():
    #Create a cluster with an outlier at a perfect 3,4,5 triangle away for easy calculation
    features = np.zeros([100,2])
    features = np.vstack([features, np.array([3,4])])
    labels = np.repeat(0, 101)
    centroids = {}
    centroids[0] = [0,0]
    distances = outlier.distance_from_centroids(features, centroids, labels)
    true_distances = [0 for x in range(100)]
    true_distances.append(5)
    
    assert distances == true_distances

    #view
    colors = [int(x > 2) for x in distances]
    visualize.plot_2d_layer(features=features, labels=colors)