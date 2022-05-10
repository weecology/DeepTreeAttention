#Year model
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch

from src.models import Hang2020

class YearModel(Module):
    def __init__(self, bands, classes, config):
        super(YearModel, self).__init__()        
        #Load from state dict of previous run
        if config["pretrain_state_dict"]:
             self.base_model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
        else:
             self.base_model = Hang2020.spectral_network(bands=config["bands"], classes=classes)
             
    def forward(self, images):
        preds = []
        for image in images:
            pred = self.base_model(image)
            preds.append(pred[-1])
        
        ensemble = torch.stack(preds, axis=1).mean(axis=1)
        
        return ensemble
        
        
        
    