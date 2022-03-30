#Hang et al. 2020 https://arxiv.org/pdf/2005.11977.pdf
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch

def global_spectral_pool(x):
    """Helper function to keep the same dimensions after pooling to avoid resizing each time"""
    global_pool = torch.mean(x,dim=(2,3))
    global_pool = global_pool.unsqueeze(-1)
    
    return global_pool
    
class conv_module(Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None):
        """Define a simple conv block with batchnorm and optional max pooling"""
        super(conv_module, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels=filters, kernel_size = (3,3), padding="same") 
        self.bn1 = nn.BatchNorm2d(filters)                    
        self.maxpool_kernal = maxpool_kernel
        if maxpool_kernel:
            self.max_pool = nn.MaxPool2d(maxpool_kernel)
        
    def forward(self, x, pool=False):
        x = self.conv_layer(x)
        x = self.bn1(x)
        x = F.relu(x)
        if pool:
            x = self.max_pool(x)
        
        return x

class vanilla_CNN(Module):
    """
    A baseline model without spectral convolutions or spatial/spectral attention 
    """
    def __init__(self, bands, classes):
        super(vanilla_CNN, self).__init__()
        self.conv1 = conv_module(in_channels=bands, filters=32)
        self.conv2 = conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
        self.conv3 = conv_module(in_channels=64, filters=128, maxpool_kernel=(2,2)) 
        # The size of the fully connected layer Assumes a certain band convo, TODO make this flexible by band number.
        self.fc1 = nn.Linear(in_features=512,out_features=classes)
    
    def forward(self, x):
        """Take an input image and run the conv blocks, flatten the output and return  features"""
        x = self.conv1(x)
        x = self.conv2(x, pool = True)
        x = self.conv3(x, pool = True)
        x = torch.flatten(x, start_dim=1)        
        x = self.fc1(x)
        
        return x

class Classifier(Module):
    """A small module to seperate the classifier head, which depends on the number of classes.
    This makes it easier to pretain on other data
    """
    def __init__(self, in_features, classes):
        super(Classifier,self).__init__()        
        self.fc1 = nn.Linear(in_features=in_features, out_features=classes)
    
    def forward(self, features):
        scores = self.fc1(features)
        
        return scores
        
class spatial_attention(Module):
    """
    Learn cross band spatial features with a set of convolutions and spectral pooling attention layers
    """
    def __init__(self, filters):
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
        
        self.attention_conv1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=kernel_size, padding="same")
        
        #Add a classfication branch with max pool based on size of the layer
        if filters == 32:
            pool_size = (4, 4)
            in_features = 128
        elif filters == 64:
            in_features = 256
            pool_size = (2, 2)
        elif filters == 128:
            in_features = 512
            pool_size = (1, 1)
        else:
            raise ValueError("Unknown filter size for max pooling")
        
        self.class_pool = nn.MaxPool2d(pool_size)
        
    def forward(self, x):
        """Calculate attention and class scores for batch"""
        #Global pooling and add dimensions to keep the same shape
        pooled_features = self.channel_pool(x)
        pooled_features = F.relu(pooled_features)
        
        #Attention layers
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        
        #Add dummy dimension to make the shapes the same
        attention = torch.mul(x, attention)
        
        # Classification Head
        pooled_attention_features = self.class_pool(attention)
        pooled_attention_features = torch.flatten(pooled_attention_features, start_dim=1)

        return attention, pooled_attention_features

class spectral_attention(Module):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    The feature maps should be pooled to remove spatial dimensions before reading in the module
    Args:
        in_channels: number of feature maps of the current image
    """
    def __init__(self, filters):
        super(spectral_attention, self).__init__()        
        # Weak Attention with adaptive kernel size based on size of incoming feature map
        if filters == 32:
            kernel_size = 3
        elif filters == 64:
            kernel_size = 5
        elif filters == 128:
            kernel_size = 7
        else:
            raise ValueError(
                "Unknown incoming kernel size {} for attention layers".format(kernel_size))
        
        self.attention_conv1 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        self.attention_conv2 = nn.Conv1d(in_channels=filters, out_channels=filters, kernel_size=kernel_size, padding="same")
        
    def forward(self, x):
        """Calculate attention and class scores for batch"""
        #Global pooling and add dimensions to keep the same shape
        pooled_features = global_spectral_pool(x)
        
        #Attention layers
        attention = self.attention_conv1(pooled_features)
        attention = F.relu(attention)
        attention = self.attention_conv2(attention)
        attention = F.sigmoid(attention)
        
        #Add dummy dimension to make the shapes the same
        attention = attention.unsqueeze(-1)
        attention = torch.mul(x, attention)
        
        # Classification Head
        pooled_attention_features = global_spectral_pool(attention)
        pooled_attention_features = torch.flatten(pooled_attention_features, start_dim=1)
        
        return attention, pooled_attention_features
    
class spatial_network(Module):
    """
        Learn spatial features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes):
        super(spatial_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, filters=32)
        self.attention_1 = spatial_attention(filters=32)
        self.classifier1 = Classifier(classes=classes, in_features=128)
    
        self.conv2 = conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
        self.attention_2 = spatial_attention(filters=64)
        self.classifier2 = Classifier(classes=classes, in_features=256)        
    
        self.conv3 = conv_module(in_channels=64, filters=128, maxpool_kernel=(2,2))
        self.attention_3 = spatial_attention(filters=128)
        self.classifier3 = Classifier(classes=classes, in_features=512)        
    
    def forward(self, x):
        """The forward method is written for training the joint scores of the three attention layers"""
        x = self.conv1(x)
        x, attention = self.attention_1(x)
        scores1 = self.classifier1(attention)
        
        x = self.conv2(x, pool = True)
        x, attention = self.attention_2(x)
        scores2 = self.classifier2(attention)
        
        x = self.conv3(x, pool = True)        
        x, attention = self.attention_3(x)
        scores3 = self.classifier3(attention)
        
        return [scores1,scores2,scores3]
    
class spectral_network(Module):
    """
        Learn spectral features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes):
        super(spectral_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, filters=32)
        self.attention_1 = spectral_attention(filters=32)
        self.classifier1 = Classifier(classes=classes, in_features=32)
    
        self.conv2 = conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
        self.attention_2 = spectral_attention(filters=64)
        self.classifier2 = Classifier(classes=classes, in_features=64)
    
        self.conv3 = conv_module(in_channels=64, filters=128, maxpool_kernel=(2,2))
        self.attention_3 = spectral_attention(filters=128)
        self.classifier3 = Classifier(classes=classes, in_features=128)
    
    def forward(self, x):
        """The forward method is written for training the joint scores of the three attention layers"""
        x = self.conv1(x)
        x, attention = self.attention_1(x)
        scores1 = self.classifier1(attention)
        
        x = self.conv2(x, pool = True)
        x, attention = self.attention_2(x)
        scores2 = self.classifier2(attention)
        
        x = self.conv3(x, pool = True)        
        x, attention = self.attention_3(x)
        scores3 = self.classifier3(attention)
        
        return [scores1,scores2,scores3]
        
class Hang2020(Module):
    def __init__(self, bands, classes):
        super(Hang2020, self).__init__()    
        self.spectral_network = spectral_network(bands, classes)
        self.spatial_network = spatial_network(bands, classes)
        
        #Learnable weight
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=float), requires_grad=True)
        
    def forward(self, x):
        spectral_scores = self.spectral_network(x)
        spatial_scores = self.spatial_network(x)
        
        #Take the final attention scores
        spectral_classes = spectral_scores[-1]
        spatial_classes = spatial_scores[-1]
        
        #Weighted average
        self.weighted_average = torch.sigmoid(self.alpha)
        joint_score = spectral_classes * self.weighted_average + spatial_classes * (1-self.weighted_average)
        
        return joint_score
        
    
def load_from_backbone(state_dict, classes, bands):
    train_state_dict = torch.load(state_dict, map_location="cpu")
    dict_items = train_state_dict.items()
    model = spectral_network(classes=classes, bands=bands)
    dict_to_update = model.state_dict()
    
    #update weights from non-classifier layers
    pretrained_dict = {k: v for k, v in dict_items if not "classifier" in k}
    dict_to_update.update(pretrained_dict)
    model.load_state_dict(dict_to_update)
    
    return model
    
    