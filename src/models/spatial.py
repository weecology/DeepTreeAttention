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
        num_classes = len(np.unique(val_labels))
        micro_recall = torchmetrics.Accuracy(average="micro")
        macro_recall = torchmetrics.Accuracy(average="macro", num_classes=len(np.unique(val_labels)))
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall,"Macro Accuracy":macro_recall})
        self.fc1 = nn.Linear(in_features=num_classes,out_features=num_classes)
        self.batch1 = nn.BatchNorm1d(num_classes)
        self.batch2 = nn.BatchNorm1d(num_classes)
        
    def forward(self, sensor_score, neighbor_score):
        neighbor_score = self.batch1(neighbor_score)        
        neighbor_score = self.fc1(neighbor_score)
        neighbor_score = F.relu(neighbor_score)
        neighbor_score = self.batch2(neighbor_score)
        skip_features = neighbor_score + sensor_score
        
        return skip_features
    
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
        
        output = self.metrics(y_hat, y) 
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
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.5,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         eps=1e-08)
                                                                 
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_loss'}