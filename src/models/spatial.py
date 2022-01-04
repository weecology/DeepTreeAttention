#Spatial model
from pytorch_lightning import LightningModule
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn
import torch
from torch import optim
import torchmetrics
import numpy as np

#Dataset class
class SpatialDataset(Dataset):
    """Dataset for Spatial Learning
    Args:
       neighbor_score: numpy array samples x classes
       sensor_score: numpy array samples x classes
    """
    def __init__(self, sensor_score, neighbor_score, labels):
        self.sensor_score = sensor_score
        self.neighbor_score = neighbor_score
        self.labels = labels

    def __len__(self):
        #0th based index
        return len(self.sensor_score)
        
    def __getitem__(self, index):
        sensor_score = self.sensor_score[index,]
        neighbor_score = self.neighbor_score[index,]
        labels = self.labels[index]
        
        return sensor_score, neighbor_score, labels
        
class spatial_fusion(LightningModule):
    """A joint fusion model of HSI sensor data and Spatial"""
    def __init__(self, train_sensor_score, train_neighbor_score, val_sensor_score, val_neighbor_score, train_labels, val_labels):
        super().__init__()
        
        self.train_ds = SpatialDataset(train_sensor_score, train_neighbor_score, train_labels)
        self.val_ds = SpatialDataset(val_sensor_score, val_neighbor_score, val_labels)
        
        #Fully connected concat learner
        self.alpha = nn.Parameter(torch.tensor(0, dtype=float), requires_grad=False)
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=len(np.unique(val_labels)))
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall,"Macro Accuracy":macro_recall})

    def forward(self, sensor_score, neighbor_score):
        self.scaled_alpha = self.alpha      
        x = sensor_score + (self.scaled_alpha * neighbor_score)
        
        return x
    
    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=32,
            num_workers=0)     
        
        return data_loader
     
    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=32,
            num_workers=0)     
        
        return data_loader
    
    
    def predict_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=32,
            num_workers=0)     
        
        return data_loader
    
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        sensor_score, neighbor_score, y = batch
        y_hat = self.forward(sensor_score, neighbor_score)
        loss = F.cross_entropy(y_hat, y)    
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        sensor_score, neighbor_score, y = batch
        y_hat = self.forward(sensor_score, neighbor_score)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss and metrics
        self.log("val_loss", loss, on_epoch=True)
        self.log("spatial_alpha", self.scaled_alpha, on_epoch=True)
        
        softmax_prob = F.softmax(y_hat, dim =1)
        output = self.metrics(softmax_prob, y) 
        self.log_dict(output)        
                    
        return loss
    
    def predict_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        sensor_score, neighbor_score, y = batch
        y_hat = self.forward(sensor_score, neighbor_score)
                    
        return y_hat
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.00001)
        
        return {'optimizer':optimizer}
