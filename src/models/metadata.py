#Metadata model
from src import models
from src import main
from torch.nn import Module
from torch.nn import functional as F
from torch import nn
import torch

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
    
class metadata_sensor_fusion(models.Hang2020):
    """A joint fusion model of HSI sensor data and metadata"""
    def __init__(self, sites, bands,classes):
        super(metadata_sensor_fusion,self).__init__()   
        
        self.metadata_model = metadata(sites, classes)
        self.sensor_model = models.Hang2020.Hang2020(bands, classes)
        
        #Fully connected concat learner
        self.fc1 = nn.Linear(in_features = classes * 2 , out_features = classes)
    
    def forward(self, images, metadata):
        metadata_softmax = self.metadata_model(metadata)
        sensor_softmax = self.sensor_model(images)
        concat_features = torch.cat([metadata_softmax, sensor_softmax])
        concat_features = self.fc1(concat_features)
        class_scores = F.softmax(concat_features)
        
        return class_scores
        
#Subclass of the training model, metadata only
class MetadataModel(main.TreeModel):
    def __init__(self, model, sites, bands,classes, label_dict):
        super(MetadataModel,self).__init__(model=model,bands=bands,classes=classes,label_dict=label_dict)  
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        inputs, y = batch
        images = inputs["HSI"]
        metadata = inputs["metadata"]
        y_hat = self.model.forward(images, metadata)
        loss = F.cross_entropy(y_hat, y)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        inputs, y = batch
        images = inputs["HSI"]   
        metadata = inputs["metadata"]
        y_hat = self.model.forward(images, metadata)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        
        output = self.metrics(y_hat, y) 
        self.log_dict(output)
        
        return loss
        
