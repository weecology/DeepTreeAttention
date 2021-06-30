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

class spatial_attention(Module):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    """
    def __init__(self, filters, classes):
        super(spatial_attention,self).__init__()
        self.channel_pool = nn.Conv2d(in_channels=filters, out_channels=1, kernel_size=1)
        
        # Weak Attention with adaptive kernel size based on size of incoming feature map
        if filters == 32:
            kernel_size = 7
        elif filters == 64:
            kernel_size = 5
        elif filters == 128:
            kernel_size = 3
        else:
            raise ValueError(
                "Unknown incoming kernel size {} for attention layers".format(kernel_size))
        
        #TOOD check padding
        self.attention_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        
        #Add a classfication branch with max pool based on size of the layer
        #TODO calculate in channels for each flattened layer
        if filters == 32:
            pool_size = (4, 4)
        elif filters == 64:
            pool_size = (2, 2)
        elif filters == 128:
            pool_size = (1, 1)
        else:
            raise ValueError("Unknown filter size for max pooling")
        
        self.class_pool = nn.MaxPool2d(pool_size)
        self.fc1 = nn.Linear(in_features=80, out_features=classes)
        
    def forward(self, x):
        x = self.channel_pool(x)
        x = F.relu(x)
        attention = self.attention_conv1(x)
        attention = F.relu(attention)
        attention = self.attention_conv2(x)
        attention = F.sigmoid(attention)
        attention = torch.mul(x, attention)
        
        pooling = self.class_pool(attention)
        pooling = torch.flatten(pooling)
        
        class_features = self.fc1(pooling)
        class_probabilities = F.softmax(class_features)
        
        return class_probabilities
        
class spatial_network(Module):
    """
        Learn spatial features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes):
        super(spatial_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, filters=32)
        self.attention_1 = spatial_attention(filters=32, classes = classes)
    
    def forward(self, x):
        features = self.conv1(x)
        features = self.attention_1(features)
        
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
        self.fc1 = nn.Linear(in_features=10, out_features=classes)
        
    def forward(self, x):
        joint_features = self.spatial_attention(x)
        joint_features = torch.flatten(joint_features)
        joint_features = self.fc1(joint_features)
        #spectral_features = self.spectral_attention(x)
        #joint_features = self.consensus_layer(spatial_features, spectral_features)
        
        return F.softmax(joint_features)

    
