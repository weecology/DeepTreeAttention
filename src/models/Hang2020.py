#Hang et al. 2020
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch

class conv_module(Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None):
        """Define a simple conv block with batchnorm and optional max pooling"""
        super(conv_module, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels=filters, kernel_size = (3,3)) 
        self.bn1 = nn.BatchNorm2d(filters)                    
        self.maxpool_kernal = maxpool_kernel
        if maxpool_kernel:
            self.max_pool = nn.MaxPool2d(maxpool_kernel)
        
    def forward(self, x):
        x = self.conv_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        return x
    
class spatial_network(Module):
    def __init__(self, bands, classes):
        super(spatial_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, filters=32)
    
    def forward(self, x):
        features = self.conv1(x)
        
        return features
    
def spectral_network(classes):
    pass

def subnetwork_consensus(classes):
    pass

class Hang2020(Module):
    def __init__(self, bands, classes):
        super(Hang2020, self).__init__()
        self.spatial_attention = spatial_network(bands=bands, classes=classes)
        #self.spectral_attention = spectral_network(classes, bands)
        #self.consensus_layer = subnetwork_consensus(classes)
        self.fc1 = nn.Linear(in_features=6146560, out_features=classes)
        
    def forward(self, x):
        joint_features = self.spatial_attention(x)
        joint_features = torch.flatten(joint_features)
        joint_features = self.fc1(joint_features)
        #spectral_features = self.spectral_attention(x)
        #joint_features = self.consensus_layer(spatial_features, spectral_features)
        
        return F.softmax(joint_features)

    
