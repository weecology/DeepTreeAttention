##Vanilla alive dead model
import pandas as pd
import comet_ml
import numpy as np
import os
import pytorch_lightning as pl
from skimage import io
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms
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

class AliveDeadDataset(Dataset):

    def __init__(self, csv_file, root_dir, label_dict = {"Alive": 0,"Dead":1}, transform=True, augment=False, train=True):
        """
        Args:
            csv_file (string): Path to a single csv file with annotations.
            root_dir (string): Directory with all the images.
            label_dict: a dictionary where keys are labels from the csv column and values are numeric labels "Tree" -> 0
        """
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_names = self.annotations.image_path.unique()
        self.label_dict = label_dict
        
        if transform is True:
            self.transform = get_transform(augment=augment)
        else:
            self.transform = None
            
        self.train = train

    def __len__(self):
        return self.annotations.shape[0]

    def __getitem__(self, idx):
        selected_row = self.annotations.loc[idx]
        img_name = os.path.join(self.root_dir, selected_row["image_path"])
        image = io.imread(img_name)

        # select annotations
        xmin, xmax, ymin, ymax = selected_row[["xmin","xmax","ymin","ymax"]].values.astype(int)
        
        xmin = np.max([0,xmin-10])
        xmax = np.min([image.shape[1],xmax+10])
        ymin = np.max([0,ymin-10])
        ymax = np.min([image.shape[0],ymax+10])
        
        box = image[ymin:ymax, xmin:xmax]
        
        if self.transform is not None:
            box = self.transform(box)
        
        # Labels need to be encoded if supplied
        if self.train:
            label = self.label_dict[selected_row.label]
            return box, label
        else:
            return box
    
#Lightning Model
class AliveDead(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)        
        self.accuracy = torchmetrics.Accuracy(average='none', num_classes=2)      
        self.total_accuracy = torchmetrics.Accuracy()        
        self.precision_metric = torchmetrics.Precision()
        
    def forward(self, x):
        output = self.model(x)
        
        return output
    
    def training_step(self, batch, batch_idx):
        x,y = batch
        outputs = self.forward(x)
        loss = F.cross_entropy(outputs,y)
        self.log("train_loss",loss)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        x,y = batch
        outputs = self(x)
        loss = F.cross_entropy(outputs,y)
        softmax_prob = F.softmax(outputs, dim =1)
        
        self.log("val_loss",loss)        
        
        self.accuracy(softmax_prob, y)
        self.total_accuracy(softmax_prob, y)
        self.precision_metric(softmax_prob, y)
        
        return softmax_prob
 
    def validation_epoch_end(self, outputs):
        alive_accuracy, dead_accuracy = self.accuracy.compute()
        self.log('alive_accuracy',alive_accuracy)
        self.log('dead_accuracy',dead_accuracy)
        self.log('val_precision', self.precision_metric.compute())
        self.log('val_acc',self.total_accuracy.compute())
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
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
    

    
