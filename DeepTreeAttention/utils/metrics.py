import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
import geopandas as gpd

def site_confusion(y_true, y_pred, site_lists):
    """What proportion of misidentified species come from the same site?
    Args: 
        y_true: string values of true labels
        y_pred: string values or predicted labels
        site_lists: list of site labels for each string label taxonID -> sites
    Returns:
        Within site confusion score
    """
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

def genus_confusion(y_true, y_pred, scientific_dict):
    """What proportion of misidentified species come from the same genus?
    Args: 
        y_true: taxonID of true labels
        y_pred: taxonID of predicted labels
        scientific_dict: a dict of taxonID -> scientific name
    Returns:
        Within site confusion score
    """
    within_genus = 0
    cross_genus = 0    
    for index, value in enumerate(y_pred):
        #If not correctly predicted
        if not value == y_true[index]:
            true_genus = scientific_dict[y_true[index]].split().str.get(0)
            pred_genus = scientific_dict[y_pred[index]].split().str.get(0)
            
            if true_genus == pred_genus:
                within_genus +=1
            else:
                cross_genus +=1
    
    #don't divide by zero
    if within_genus + cross_genus == 0:
        return 0
    
    #Get proportion of within site error
    proportion_within = within_genus/(within_genus + cross_genus)
    
    return proportion_within

        
def f1_scores(y_true, y_pred):
    """Calculate micro, macro
    Args:
        y_true: one_hot ground truth labels
        y_pred: softmax classes 
    Returns:
        macro: macro average fscore
        micro: micro averge fscore
    """
    #F1 scores
    y_true_integer = np.argmax(y_true, axis=1)
    y_pred_integer = np.argmax(y_pred, axis=1)

    macro = f1_score(y_true_integer, y_pred_integer, average='macro')
    micro = f1_score(y_true_integer, y_pred_integer, average='micro')

    return macro, micro


def confusion(y_true, y_pred, num_classes):
    confusion = tf.math.confusion_matrix(labels=y_true,
                                         predictions=y_pred,
                                         num_classes=num_classes)
    return confusion
