#Autoencoder
from torch.nn import Module
from torch.nn import functional as F
import torch.nn as nn
import torch
from src.models. Hang2020 import conv_module
        
class autoencoder(Module):
    def __init__(self, bands, classes):
        super(autoencoder, self).__init__()    
        self.conv1 = conv_module(in_channels=bands, filters=32)    
        self.conv2 = conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
        self.conv3 = conv_module(in_channels=64, filters=128, maxpool_kernel=(2,2))
        self.fc1 = nn.Linear(in_features=128, out_features=classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        
        return x