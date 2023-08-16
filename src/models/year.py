#Year model
from torch import nn
import torch
from src.models import Hang2020

class learned_ensemble(nn.Module):
    def __init__(self, years, classes, config):
        super().__init__()
        self.year_models = nn.ModuleDict()
        for year in years:
            #Load from state dict of previous run
            if config["pretrain_state_dict"]:
                base_model = Hang2020.load_from_backbone(state_dict=config["pretrain_state_dict"], classes=classes, bands=config["bands"])
            else:
                base_model = Hang2020.Single_Spectral_Model(bands=config["bands"], classes=classes)
            
            self.year_models[str(year)] = base_model
        
    def forward(self, images):
        year_scores = []
        for year, image in images.items():
            # Skip padding tensors
            if image.sum() == 0:
                continue
            try:
                score = self.year_models[year](image)
            except KeyError:
                continue
            year_scores.append(score)
        if len(year_scores) == 0:
            return None
        return torch.stack(year_scores, axis=1).mean(axis=1)
                
    
