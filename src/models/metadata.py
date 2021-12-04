#Metadata model
from src.models.Hang2020 import Hang2020
from src import main
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch

class metadata(Module):
    def __init__(self, sites, classes):
        super(metadata,self).__init__()    
        self.embedding = nn.Embedding(sites, 12)
        #self.dropout = nn.Dropout(p=0.7)
        self.batch_norm = nn.BatchNorm1d(12)   
        #self.mlp = nn.Linear(in_features=12, out_features=classes)
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.batch_norm(x)        
        #x = self.dropout(x)           
        #x = self.mlp(x)
        
        return x
    
class metadata_sensor_fusion(Module):
    """A joint fusion model of HSI sensor data and metadata"""
    def __init__(self, bands, sites, classes):
        super(metadata_sensor_fusion,self).__init__()   
        
        self.metadata_model = metadata(sites, classes)
        self.sensor_model = Hang2020(bands, classes)
                
        #Fully connected concat learner
        self.fc1 = nn.Linear(in_features = classes * 2 , out_features = classes)
    
    def forward(self, images, metadata):
        metadata_softmax = self.metadata_model(metadata)
        sensor_softmax = self.sensor_model(images)
        concat_features = torch.cat([metadata_softmax, sensor_softmax], dim=1)
        concat_features = self.fc1(concat_features)
        concat_features = F.relu(concat_features)
        
        return concat_features
        
#Subclass of the training model, metadata only
class MetadataModel(main.TreeModel):
    """Subclass the core model and update the training and val loop to take two inputs"""
    def __init__(self, model,classes, label_dict, config):
        super(MetadataModel,self).__init__(model=model,classes=classes,label_dict=label_dict, config=config)  
    
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]
        metadata = inputs["site"]
        y_hat = self.model.forward(images, metadata)
        
        loss = F.cross_entropy(y_hat, y)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        individual, inputs, y = batch
        images = inputs["HSI"]   
        metadata = inputs["site"]
        
        y_hat = self.model.forward(images, metadata)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        
        if not self.training:
            y_hat = F.softmax(y_hat, dim = 1)
            
        output = self.metrics(y_hat, y) 
        self.log_dict(output)
        
        return loss
        
    def predict(self, inputs):
        feature = self.model(inputs["HSI"], inputs["site"])
        return F.softmax(feature, dim=1)
    