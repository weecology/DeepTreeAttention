#Outlier detection module
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from src import visualize
from src import data
import scipy
import tempfile
import torch

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
        corruption_precision = sum(corrupted_data.predicted_outlier)/sum(results.predicted_outlier)

    true_outliers = results[~(results.label == results.observed_label)]
    
    #inset data does not have class 8 ir 9
    novel = true_outliers[true_outliers.label.isin([8,9])]
    if novel.empty:
        novel_accuracy = None
    else:
        novel_accuracy = sum(novel.predicted_outlier)/novel.shape[0]
        
    if true_outliers.empty:
        return pd.DataFrame({"autoencoder_label_switching_accuracy": [None], "autoencoder_label_switching_precision": [None], "classification_accuracy": [mean_accuracy]})
    else:
        #inset data does not have class 8 ir 9
        inset = true_outliers[~true_outliers.label.isin([8,9])]
        outlier_accuracy = sum(inset.predicted_outlier)/inset.shape[0]
        outlier_precision = sum(inset.predicted_outlier)/sum(results[~results.label.isin([8,9])].predicted_outlier)
        
        if experiment:
            experiment.log_metric("autoencoder_novel_accuracy", novel_accuracy)            
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
    #Calculate mean and covariance matrix within feature space
    centroids = calculate_centroids(features, labels)
    cov = calculate_covariance(features, labels)
    results["centroid_distance"] = distance_from_centroids(features, centroids, labels, cov)
    results["distance_outlier"] = results["centroid_distance"] > threshold
    
    true_outliers = results[~(results.label == results.observed_label)]
    
    if true_outliers.empty:
        return pd.DataFrame({"distance_outlier_accuracy": [None], "distance_outlier_precision": [None]})
    
    #inset data does not have class 8 ir 9
    novel = true_outliers[true_outliers.label.isin([8,9])]
    if novel.empty:
        novel_accuracy = None
    else:
        novel_accuracy = sum(novel.distance_outlier)/novel.shape[0]
    
    #Label switching
    true_outliers = results[~(results.label == results.observed_label)]
    inset = true_outliers[~true_outliers.label.isin([8,9])]    
    outlier_accuracy = sum(inset.distance_outlier)/inset.shape[0]
    try:
        outlier_precision = sum(inset.distance_outlier)/sum(results[~results.label.isin([8,9])].distance_outlier)
    except:
        outlier_precision = None
    
    #Image corruptions
    corrupted_data = results[results.image_corrupt==True]
    if corrupted_data.empty:
        corruption_accuracy = None
        corruption_precision = None
    else:     
        corruption_accuracy = sum(corrupted_data.distance_outlier)/corrupted_data.shape[0]
        try:
            corruption_precision = sum(corrupted_data.distance_outlier)/sum(results.distance_outlier)
        except:
            corruption_precision = None
    
    if experiment:        
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
            
        experiment.log_metric("distance_novel_accuracy", novel_accuracy)
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
    
def predict_outliers(model, annotations, config, comet_logger=None):
    """Predict outliers and append a column to annotations based on a given model. Model object must contain a predict method"""
    
    #predict annotations
    with tempfile.TemporaryDirectory() as tmpdir:
        annotations.to_csv("{}/annotations.csv".format(tmpdir))
        ds = data.TreeDataset(csv_file="{}/annotations.csv".format(tmpdir), image_size=config["image_size"])
        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=config["batch_size"],
            num_workers=config["workers"],
            shuffle=False
        )
        
        results = model.predict(data_loader)
    features = model.classification_bottleneck

    # distance outlier
    centroids = calculate_centroids(features, results.observed_label)
    cov = calculate_covariance(features, results.observed_label)
    results["centroid_distance"] = distance_from_centroids(features, centroids, results.observed_label, cov)
    
    #loss outlier
    threshold = results.autoencoder_loss.quantile(config["outlier_threshold"])    
    results["predicted_outlier"] = (results["centroid_distance"] > config["distance_threshold"]) | (results.autoencoder_loss > threshold)
    annotations["predicted_outlier"] = results["predicted_outlier"].values
    
    #Plot outliers
    #plot historgram
    if comet_logger:
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(results.autoencoder_loss, bins=20, color='c', edgecolor='k', alpha=0.65)
        ax.axvline(threshold, color='k', linestyle='dashed', linewidth=1)    
        comet_logger.experiment.log_figure(figure_name="autoencoder_loss")        
        
        #plot encoder set
        layerplot_vis = visualize.plot_2d_layer(features=features, labels=results.observed_label, use_pca=False)
        comet_logger.experiment.log_figure(figure=layerplot_vis, figure_name="classification_bottleneck_labels")        
        
    return annotations
    
    