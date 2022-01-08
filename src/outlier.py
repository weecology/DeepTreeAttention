#Outlier detection module
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from src import visualize
from src import data
import scipy
import tempfile
import torch
import os
from sklearn.neighbors import LocalOutlierFactor

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
        experiment.log_parameter("Autoencoder quantile", threshold)
    
    results["predicted_outlier"] = results.autoencoder_loss > threshold
    
    if experiment:
        experiment.log_table("results.csv",results)
    
    #Image corruptions
    corrupted_data = results[results.outlier=="corrupted"]
    inset = results[results.outlier=="inlier"]
    
    if corrupted_data.empty:
        corruption_accuracy = None
        corruption_precision = None
    else:     
        corruption_accuracy = sum(corrupted_data.predicted_outlier)/corrupted_data.shape[0]
        corruption_precision = sum(corrupted_data.predicted_outlier)/(sum(inset.predicted_outlier) + sum(corrupted_data.predicted_outlier))

        if experiment:
            experiment.log_metric("autoencoder_image_corruption_accuracy", corruption_accuracy)
            experiment.log_metric("autoencoder_image_corruption_precision", corruption_precision)            
        
        return pd.DataFrame({"autoencoder_image_corruption_accuracy":[corruption_accuracy],
                             "autoencoder_image_corruption_precision":[corruption_precision]})
                
def distance_outliers(results, features, labels, threshold, experiment):
    """Detect clusters and identify mislabeled data
    Args:
        results: predictions from the validation set as pandas dataframe
        features: numpy array of test set features
        labels: class of each test set, in the same order as feature array first dimension
        threshold: numeric threshold for outlier delination from class center in feature space
        """
    #Calculate mean and covariance matrix within feature space
    centroids = calculate_centroids(features, labels)
    cov = calculate_covariance(features, labels)
    results["centroid_distance"] = distance_from_centroids(features, centroids, labels, cov)
    distance_quantile = results.centroid_distance.quantile(threshold)
    results["distance_outlier"] = results["centroid_distance"] > distance_quantile
    
    label_swap = results[results.outlier == "label_swap"]
    inset = results[results.outlier == "inlier"]
    
    if label_swap.empty:
        return pd.DataFrame({"distance_outlier_accuracy": [None], "distance_outlier_precision": [None]})
    
    #Label switching
    outlier_accuracy = sum(label_swap.distance_outlier)/label_swap.shape[0]
    try:
        outlier_precision = sum(label_swap.distance_outlier)/(sum(inset.distance_outlier) + sum(label_swap.distance_outlier))
    except:
        outlier_precision = None
    
    if experiment:        
        experiment.log_parameter("distance_quantile",distance_quantile)
        #Distance by outlier type
        box = results.boxplot("distance_outlier", by="outlier", return_type='axes')
        box[0].set_ylabel("Distance from class centroid")
        box[0].set_xlabel("Type of outlier")
        
        #Plot centroid distance
        for x in centroids:
            fig = plt.figure()
            ax = plt.subplot()
            cluster_points = results[results.observed_label == x]            
            cluster_points["centroid_distance"].hist(ax=ax)
            experiment.log_figure("class {} centroid distance".format(x))
            plt.close(fig)
            
            centroid_plot = visualize.plot_2d_layer(features[results.observed_label == x,:], labels=cluster_points["distance_outlier"].astype(int), size_weights=cluster_points["distance_outlier"].astype(int)+1)
            plt.plot(centroids[x][0], centroids[x][1],'go')
            plt.title("Class {} predicted outliers".format(x))
            experiment.log_figure("class {} predicted outliers".format(x))      
            
            cluster_points["label_swap"] = cluster_points["outlier"] == "label_swap"
            centroid_plot = visualize.plot_2d_layer(features[results.observed_label == x,:], labels=cluster_points["label_swap"].astype(int), size_weights=cluster_points["distance_outlier"].astype(int)+1)
            plt.plot(centroids[x][0], centroids[x][1],'go')
            plt.title("Class {} true outliers".format(x))
            experiment.log_figure("class {} true outliers".format(x))      
            
        experiment.log_metric("distance_label_switching_accuracy", outlier_accuracy)
        experiment.log_metric("distance_label_switching_precision", outlier_precision)
        
    return pd.DataFrame({"distance_label_switching_accuracy": [outlier_accuracy], "distance_label_switching_precision": [outlier_precision]})
    
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

def calculate_covariance(features, labels):
    """calculate class centroids in a multidim feature space
    Args:
        features: a numpy array of B, C, H, W, usually a encoding feature block
        labels: class label of each sample in the same order as the feature block batch dimensions
    Returns:
        centroids: a dict of locations for each class
    """
    unique_labels = np.unique(labels)
    cov = {}
    for x in unique_labels:
        class_features = features[labels == x,:]
        try: 
            cov[x] = scipy.linalg.inv(np.cov(class_features.T))
        except:
            cov[x] = np.array([[0,0],[0,0]])
        
    return cov

def distance_from_centroids(features, centroids, labels, cov):
    """Euclidean distance from feature center for each label
    Args:
        features: numpy array of features to cluster
        centroids: dictionary of centroid features class -> numpy_array
        labels: list of classes in the order of feature rows
        cov: dictionary of var-cov matrix of each feature class
    """
    distances = []
    for x in range(features.shape[0]):  
        mahal_dist = scipy.spatial.distance.mahalanobis(features[x,:], centroids[labels[x]], cov[labels[x]])        
        distances.append(mahal_dist)
    
    return distances
    
def novel_detection(results, features, experiment):
    """Novel individual detection using projection layer features
    Args:
        results: result dataframe, see main.predict_dataloader
        features: array of projection features
    """
    lof = LocalOutlierFactor()
    y_pred = lof.fit_predict(features)    
    results["predicted_novel"] =  y_pred==-1
    novel = results[results["outlier"] == "novel"]
    inlier = results[results["outlier"] == "inlier"]
    
    #Recall
    novel_recall = np.sum(novel["predicted_novel"])/novel.shape[0]
    
    #Precision
    novel_precision = np.sum(novel["predicted_novel"])/(np.sum(inlier["predicted_novel"]) + np.sum(novel["predicted_novel"]))
    
    if experiment:
        experiment.log_metric("Novel Recall", novel_recall)
        experiment.log_metric("Novel Precision", novel_precision)
    
    return pd.DataFrame({"novel_recall": [novel_recall], "novel_precision": [novel_precision]})
