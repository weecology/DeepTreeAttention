#RGB model
from src.models.Hang2020 import *
from torch.nn import Module
from torch.nn import functional as F
from torch import nn

class RGB_conv_module(Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None):
        """Define a simple conv block with batchnorm and optional max pooling"""
        super(RGB_conv_module, self).__init__()
        self.conv_layer = nn.Conv2d(in_channels, out_channels=filters, kernel_size = (3,3), stride=3) 
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


class RGB(nn.Module):
    def __init__(self):
        super(RGB, self).__init__()        
        self.conv1 = RGB_conv_module(in_channels=3, filters=16)
        self.conv2 = RGB_conv_module(in_channels=16, filters=32)
        self.conv3 = RGB_conv_module(in_channels=32, filters=64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        return x

class fusion_attention(Module):
    """
    Learn cross band spectral features with a set of convolutions and spectral pooling attention layers
    The feature maps should be pooled to remove spatial dimensions before reading in the module
    Args:
        in_channels: number of feature maps of the current image
    """
    def __init__(self, filters):
        super(fusion_attention, self).__init__()        
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
    
class spectral_fusion_network(Module):
    """
        Learn spectral features with alternating convolutional and attention pooling layers
    """
    def __init__(self, bands, classes):
        super(spectral_fusion_network, self).__init__()
        
        #First submodel is 32 filters
        self.conv1 = conv_module(in_channels=bands, filters=32)
        self.attention_1 = fusion_attention(filters=32)
    
        self.conv2 = conv_module(in_channels=32, filters=64, maxpool_kernel=(2,2))
        self.RGB_features = RGB()
        self.attention_2 = fusion_attention(filters=128)
    
        self.conv3 = conv_module(in_channels=128, filters=128, maxpool_kernel=(2,2))
        self.attention_3 = fusion_attention(filters=128)
        self.classifier3 = Classifier(classes=classes, in_features=128)
    
    def forward(self, hsi_image, rgb_image):
        """The forward method is written for training the joint scores of the three attention layers"""
        x = self.conv1(hsi_image)
        x, attention = self.attention_1(x)
        
        x = self.conv2(x, pool = True)
        rgb_features = self.RGB_features(rgb_image)
        
        #Upsample to ensure the maps are the same size.
        rgb_features = torch.nn.Upsample(size=x.shape[-2:], mode="bilinear")(rgb_features)
        x = torch.cat([x, rgb_features], dim=1)
        x, attention = self.attention_2(x)
        
        x = self.conv3(x, pool = True)        
        x, attention = self.attention_3(x)
        scores = self.classifier3(attention)
        
        return scores
    
def load_from_backbone(state_dict, classes, bands):
    train_state_dict = torch.load(state_dict, map_location="cpu")
    dict_items = train_state_dict.items()
    model = spectral_fusion_network(classes=classes, bands=bands)
    dict_to_update = model.state_dict()
    
    #update weights from non-classifier layers
    pretrained_dict = {k: v for k, v in dict_items if not "classifier" in k}
    dict_to_update.update(pretrained_dict)
    model.load_state_dict(dict_to_update)
    
    return model