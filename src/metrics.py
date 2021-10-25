#Metrics
import torch
import numpy as np
import pandas as pd
from torch.nn import functional as F
from src import data

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
            incorrect_site = site_lists[y_pred[index]]
        
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
            true_genus = scientific_dict[y_true[index]][0].split()[0]
            pred_genus = scientific_dict[y_pred[index]][0].split()[0]
            
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

def novel_prediction(model, csv_file, config):
    """Predict a dataset of species not included in the dataset and get the final activation score before/after softmax"""
    novel_ds = data.TreeDataset(csv_file, image_size=config["image_size"], config=config)
    
    data_loader = torch.utils.data.DataLoader(
        novel_ds,
        batch_size=config["batch_size"],
        num_workers=config["workers"])
    
    model.eval()
    top_scores = []
    softmax_scores = []
    individuals = []
    for batch in data_loader:
        individual, inputs, targets = batch
        with torch.no_grad():
            pred = model.predict(inputs)
            top_score = pred[np.argmax(pred)]
            softmax_pred = F.softmax(pred, dim=0)
        top_scores.append(top_score)
        softmax_scores.append(softmax_pred)
            
    top_scores = np.concatenate(top_scores)              
    softmax_scores = np.concatenate(softmax_scores)  
    features = pd.DataFrame({"individual":individuals, "top_score": top_scores,"softmax_score":softmax_scores})
    
    return features    