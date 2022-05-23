#Year model
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch
import torchmetrics
from src.models import Hang2020
import numpy as np
import pandas as pd

class learned_ensemble(Module):
    def __init__(self, classes, config):
        super().__init__()
        #Load from state dict of previous run
        if config["pretrain_state_dict"]:
            self.base_model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
        else:
            self.base_model = Hang2020.spectral_network(bands=config["bands"], classes=classes)
        
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=classes)
        self.metrics = torchmetrics.MetricCollection(
            {"Micro Accuracy":micro_recall,
             "Macro Accuracy":macro_recall,
             })
        
    def forward(self,images):
        year_scores = []
        for x in images:
            x = self.base_model(x)
            score = x[-1]            
            year_scores.append(score)
        
        return torch.stack(year_scores, axis=1).mean(axis=1)
                
    
