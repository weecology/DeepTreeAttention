##Vanilla alive dead model
import pandas as pd
import comet_ml
import numpy as np
import os
import pytorch_lightning as pl
import rasterio
from skimage import io
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import torchmetrics

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def get_transform(augment):
    data_transforms = []
    data_transforms.append(transforms.ToTensor())
    data_transforms.append(normalize)
    data_transforms.append(transforms.Resize([224,224]))
    if augment:
        data_transforms.append(transforms.RandomHorizontalFlip(0.5))
    return transforms.Compose(data_transforms)
    
#Lightning Model
class AliveDead(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        
        # Model
        self.model = models.resnet50(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = torch.nn.Linear(num_ftrs, 2)        
        
        # Metrics
        self.accuracy = torchmetrics.Accuracy(average='none', num_classes=2, task="multiclass")      
        self.total_accuracy = torchmetrics.Accuracy(num_classes=2, task="multiclass")        
        self.precision_metric = torchmetrics.Precision(num_classes=2, task="multiclass")
        self.metrics = torchmetrics.MetricCollection({"Class Accuracy":self.accuracy, "Accuracy":self.total_accuracy, "Precision":self.precision_metric})
        
        # Data
        self.config = config
        self.ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        train_dir = os.path.join(self.ROOT,config["dead"]["train_dir"])
        val_dir = os.path.join(self.ROOT,config["dead"]["test_dir"])
        self.train_ds = ImageFolder(root=train_dir, transform=get_transform(augment=True))
        self.val_ds = ImageFolder(root=val_dir, transform=get_transform(augment=False))
        
    def forward(self, x):
        output = self.model(x)
        output = F.sigmoid(output)

        return output
    
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=self.config["dead"]["batch_size"],
            shuffle=True,
            num_workers=self.config["dead"]["num_workers"]
        )   
        
        return train_loader
    
    def predict_dataloader(self, ds):
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=self.config["dead"]["batch_size"],
            shuffle=False,
            num_workers=self.config["dead"]["num_workers"]
        )   
        
        return loader
    
    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=self.config["dead"]["batch_size"],
            shuffle=True,
            num_workers=self.config["dead"]["num_workers"]
        )   
        
        return val_loader
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs,y)
        self.log("train_loss",loss)
        
        return loss
      
    def predict_step(self, batch, batch_idx):
        outputs = self.forward(batch)
        yhat = F.softmax(outputs, 1)
        
        return yhat
    
    def validation_step(self, batch, batch_idx):
        x,y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)        
        self.log("val_loss",loss)      
        metric_dict = self.metrics(outputs, y)
        self.log("Alive Accuracy",metric_dict["Class Accuracy"][0])
        self.log("Dead Accuracy",metric_dict["Class Accuracy"][1])        
        #self.log_dict(metric_dict)
        
        return loss
    
    def on_validation_epoch_end(self):
        val_metrics = self.metrics.compute()
        self.log("Alive Accuracy",val_metrics["Class Accuracy"][0])
        self.log("Dead Accuracy",val_metrics["Class Accuracy"][1])  
        self.log("Accuracy",val_metrics["Accuracy"])
        self.log("Accuracy",val_metrics["Precision"])
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config["dead"]["lr"])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    mode='min',
                                                                    factor=0.5,
                                                                    patience=10,
                                                                    verbose=True,
                                                                    threshold=0.0001,
                                                                    threshold_mode='rel',
                                                                    cooldown=0,
                                                                    min_lr=0,
                                                                    eps=1e-08)
        
        #Monitor rate is val data is used
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_loss'}
            
    def dataset_confusion(self, loader):
        """Create a confusion matrix from a data loader"""
        true_class = []
        predicted_class = []
        self.eval()
        for batch in loader:
            x,y = batch
            true_class.append(F.one_hot(y,num_classes=2).detach().numpy())
            prediction = self(x)
            predicted_class.append(prediction.detach().numpy())
        
        true_class = np.concatenate(true_class)
        predicted_class = np.concatenate(predicted_class)

        return true_class, predicted_class

class utm_dataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       crowns: geodataframe of crown locations from a single rasterio src
       preload: there is only 1 RGB tile, load it just once on init
    """
    def __init__(self, crowns, config=None, preload=True):
        self.config = config 
        self.crowns = crowns
        self.image_size = config["image_size"]
        self.transform = get_transform(augment=False)
        image_path = self.crowns.RGB_tile.iloc[0]        
        self.RGB_src = rasterio.open(image_path)
 
    def __len__(self):
        #0th based index
        return self.crowns.shape[0]
        
    def __getitem__(self, index):
        #Load crown and crop RGB
        geom = self.crowns.iloc[index].geometry
        left, bottom, right, top = geom.bounds
        box = self.RGB_src.read(window=rasterio.windows.from_bounds(left-1, bottom-1, right+1, top+1, transform=self.RGB_src.transform))             
        
        # Channels last
        box = np.rollaxis(box,0,3)
        
        # Preprocess
        image = self.transform(box)
            
        return image

def index_to_example(index, test_dataset):
    image_array = test_dataset[index][0].numpy()
    image_array = np.rollaxis(image_array, 0,3)
    image_name = "confusion-matrix-%05d.png" % index
    results = experiment.log_image(
        image_array, name=image_name,
    )
    # Return sample, assetId (index is added automatically)
    return {"sample": image_name, "assetId": results["imageId"]}