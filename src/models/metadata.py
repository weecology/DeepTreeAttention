#Metadata model
from torch.nn import Module
from torch.nn import functional as F
from torch import nn

class metadata(Module):
    def __init__(self, sites, classes):
        super(metadata,self).__init__()    
        self.mlp = nn.Linear(in_features=sites, out_features=classes)
        self.bn = nn.BatchNorm1d(num_features=classes)
    
    def forward(self, x):
        x = self.mlp(x)
        x = F.relu(x)
        x = self.bn(x)
        x = F.softmax(x)
        
        return x