#Outlier detection module
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from src import visualize

def autoencoder_outliers(results, outlier_threshold, experiment):
    """Given a set of predictions, label outliers"""
    threshold = results.autoencoder_loss.quantile(outlier_threshold)
        
    #plot historgram
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(results.autoencoder_loss, bins=20, color='c', edgecolor='k', alpha=0.65)
    ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)
    
    if experiment:
        experiment.log_figure(figure_name="loss_histogram", figure=fig)
    
    fig = plt.figure()
    ax = fig.add_subplot()        
    props = dict(boxes="Gray", whiskers="Orange", medians="Blue", caps="Gray")        
    box = results.boxplot("autoencoder_loss", by="outlier", patch_artist=True, color=props, ax=ax)
    ax.axhline(threshold, color='k', linestyle='dashed', linewidth=1)        
    
    if experiment:
        experiment.log_figure(figure_name="outlier_boxplots")
    
    print("Reconstruction threshold is {}".format(threshold))
    results["predicted_outlier"] = results.autoencoder_loss > threshold
    
    if experiment:
        experiment.log_table("results.csv",results)
    
    #Mean Proportion of true classes are correct
    mean_accuracy = results[~results.label.isin([8,9])].groupby("label").apply(lambda x: x.label == x.predicted_label).mean()
    
    if experiment:
        experiment.log_metric(name="Classification accuracy", value=mean_accuracy)
    
    #Image corruptions
    corrupted_data = results[results.image_corrupt==True]
    if corrupted_data.empty:
        corruption_accuracy = None
        corruption_precision = None
    else:     
        corruption_accuracy = sum(corrupted_data.predicted_outlier)/corrupted_data.shape[0]
        corruption_precision = sum(corrupted_data.predicted_outlier)/results.shape[0]

    true_outliers = results[~(results.label == results.observed_label)]
    
    if true_outliers.empty:
        return pd.DataFrame({"autoencoder_label_switching_accuracy": [None], "autoencoder_label_switching_precision": [None], "classification_accuracy": [mean_accuracy]})
    else:
        #inset data does not have class 8 ir 9
        inset = true_outliers[~true_outliers.label.isin([8,9])]
        outlier_accuracy = sum(inset.predicted_outlier)/inset.shape[0]
        outlier_precision = sum(inset.predicted_outlier)/results.filter(~results.label.isin([8,9])).shape[0]
        
        if experiment:
            experiment.log_metric("autoencoder_label_switching_accuracy", outlier_accuracy)
            experiment.log_metric("autoencoder_label_switching_precision", outlier_precision)
            experiment.log_metric("autoencoder_image_corruption_accuracy", corruption_accuracy)
            experiment.log_metric("autoencoder_image_corruption_precision", corruption_precision)            
        return pd.DataFrame({"autoencoder_label_switching_accuracy": [outlier_accuracy],
                             "autoencoder_label_switching_precision": [outlier_precision],
                             "classification_accuracy": [mean_accuracy],
                             "autoencoder_image_corruption_accuracy":[corruption_accuracy],
                             "autoencoder_image_corruption_precision":[corruption_precision]})
                
def distance_outliers(results, features, labels, threshold, experiment):
    """Detect clusters and identify mislabeled data
    Args:
        results: predictions from the validation set as pandas dataframe
        features: numpy array of test set features
        labels: class of each test set, in the same order as feature array first dimension
        threshold: numeric threshold for outlier delination from class center in feature space
        """
    centroids = calculate_centroids(features, labels)
    results["centroid_distance"] = distance_from_centroids(features, centroids, labels)
    results["distance_outlier"] = results["centroid_distance"] > threshold
    
    true_outliers = results[~(results.label == results.observed_label)]
    
    if true_outliers.empty:
        return pd.DataFrame({"distance_outlier_accuracy": [None], "distance_outlier_precision": [None]})
    
    #inset data does not have class 8 ir 9
    novel = true_outliers[true_outliers.label.isin([8,9])]
    if novel.empty:
        novel_accuracy = None
    else:
        novel_accuracy = sum(novel.outlier)/novel.shape[0]
    
    #Label switching
    true_outliers = results[~(results.label == results.observed_label)]
    inset = true_outliers[~true_outliers.label.isin([8,9])]    
    outlier_accuracy = sum(inset.predicted_outlier)/inset.shape[0]
    outlier_precision = sum(inset.predicted_outlier)/results.filter(~results.label.isin([8,9])).shape[0]    
    
    #Image corruptions
    corrupted_data = results[results.image_corrupt==True]
    if corrupted_data.empty:
        corruption_accuracy = None
        corruption_precision = None
    else:     
        corruption_accuracy = sum(corrupted_data.distance_outlier)/corrupted_data.shape[0]
        corruption_precision = sum(corrupted_data.distance_outlier)/results.shape[0]
    
    if experiment:        
        #Distance by outlier type
        box = results.boxplot("distance_outlier", by="outlier", return_type='axes')
        box[0].set_ylabel("Distance from class centroid")
        box[0].set_xlabel("Type of outlier")
        
        #Plot centroid distance
        for x in centroids:
            cluster_points = results[results.observed_label == x]
            centroid_plot = visualize.plot_2d_layer(features[results.observed_label == x,:], labels=cluster_points["distance_outlier"].astype(int), size_weights=cluster_points["distance_outlier"].astype(int)+1)
            plt.plot(centroids[x][0], centroids[x][1],'go')
            plt.title("Class {} predicted outliers".format(x))
            experiment.log_figure("class {} predicted outliers".format(x))      
            
            cluster_points["label_swap"] = cluster_points["outlier"] == "label_swap"
            centroid_plot = visualize.plot_2d_layer(features[results.observed_label == x,:], labels=cluster_points["label_swap"].astype(int), size_weights=cluster_points["distance_outlier"].astype(int)+1)
            plt.plot(centroids[x][0], centroids[x][1],'go')
            plt.title("Class {} true outliers".format(x))
            experiment.log_figure("class {} true outliers".format(x))      
            
        experiment.log_metric("novel_accuracy", novel_accuracy)
        experiment.log_metric("distance_label_switching_accuracy", outlier_accuracy)
        experiment.log_metric("distance_label_switching_precision", outlier_precision)
        experiment.log_metric("distance_corruption_accuracy", corruption_accuracy)
        experiment.log_metric("distance_corruption_precision", corruption_precision)
        
    return pd.DataFrame({"distance_label_switching_accuracy": [outlier_accuracy], "distance_label_switching_precision": [outlier_precision], "novel_accuracy":[novel_accuracy]})
    
def calculate_centroids(features, labels):
    """calculate class centroids in a multidim feature space
    Args:
        features: a numpy array of B, C, H, W, usually a encoding feature block
        labels: class label of each sample in the same order as the feature block batch dimensions
    Returns:
        centroids: a dict of locations for each class
    """
    unique_labels = np.unique(labels)
    centroids = {}
    for x in unique_labels:
        class_features = features[labels == x,:]
        centroids[x] = class_features.mean(axis=0)
        
    return centroids
        
def distance_from_centroids(features, centroids, labels):
    """Euclidean distance from feature center for each label
    """
    distances = []
    for x in range(features.shape[0]):      
        distance = np.linalg.norm(features[x,:] - centroids[labels[x]])
        distances.append(distance)
    
    return distances
    
