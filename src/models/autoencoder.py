#Autoencoder module for HSI dimensionality reduction
#Autoencoder
from torch.utils.data import Dataset
from torch.nn import functional as F
import os
import torch.nn as nn
import torch
from torch import optim
from src.models import Hang2020
from src.models.Hang2020 import conv_module
from src import augmentation
from src import utils
from pytorch_lightning import LightningModule
import torchmetrics

#Dataset class
class AutoencoderDataset(Dataset):
    """A csv file with a path to image crop and label
    Args:
       csv_file: path to csv file with image_path and label
       df: a pandas dataframe with image_path column
    """
    def __init__(self, df, image_size=10, config=None):
        self.annotations = df
        self.config = config 
        if self.config:
            self.image_size = config["image_size"]
        else:
            self.image_size = image_size
        
        #Create augmentor
        self.transformer = augmentation.train_augmentation(image_size=image_size)
        
    def __len__(self):
        #0th based index
        return self.annotations.shape[0]
        
    def __getitem__(self, index):
        image_path = self.annotations.image_path.loc[index]
        image_path = os.path.join(self.config["crop_dir"],image_path)            
        image = utils.load_image(image_path, image_size=self.image_size)
    
        return image
    
class encoder_block(nn.Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None, pool=False):
        super(encoder_block, self).__init__()
        self.conv = conv_module(in_channels, filters)
        self.bn = nn.BatchNorm2d(num_features=filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x

class decoder_block(nn.Module):
    def __init__(self, in_channels, filters, maxpool_kernel=None, pool=False):
        super(decoder_block, self).__init__()
        self.conv = conv_module(in_channels, filters)
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=filters, kernel_size=(2,2))
        self.bn = nn.BatchNorm2d(num_features=filters)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)

        return x
    
class autoencoder(LightningModule):
    def __init__(self, train_df, val_df, classes, config, comet_logger):
        super(autoencoder, self).__init__()    
        
        self.config = config
        self.comet_logger = comet_logger
        
        #Encoder
        self.encoder_block1 = encoder_block(in_channels=config["bands"], filters=config["autoencoder_depth"]*3, pool=True)
        self.encoder_block2 = encoder_block(in_channels=config["autoencoder_depth"]*3, filters=config["autoencoder_depth"]*2, pool=True)
        self.encoder_block3 = encoder_block(in_channels=config["autoencoder_depth"]*2, filters=config["autoencoder_depth"], pool=True)
                
        #Decoder
        self.decoder_block1 = decoder_block(in_channels=config["autoencoder_depth"], filters=config["autoencoder_depth"]*2)
        self.decoder_block2 = decoder_block(in_channels=config["autoencoder_depth"]*2, filters=config["autoencoder_depth"]*3)
        self.decoder_block3 = decoder_block(in_channels=config["autoencoder_depth"]*3, filters=config["bands"])
        
        #Metrics
        mse = torchmetrics.MeanSquaredError()
        self.metrics = torchmetrics.MetricCollection({"Mean Squared Error":mse}, prefix="autoencoder_")
        
        self.train_ds = AutoencoderDataset(df=train_df, config=config)
        self.val_ds = AutoencoderDataset(df=val_df, config=config)
    
    def train_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.train_ds,
            shuffle=True,
            batch_size=self.config["autoencoder_batch_size"],
            num_workers=0)     

        return data_loader

    def val_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.config["autoencoder_batch_size"],
            num_workers=0)     

        return data_loader
    
    def predict_dataloader(self):
        data_loader = torch.utils.data.DataLoader(
            self.val_ds,
            shuffle=False,
            batch_size=self.config["autoencoder_batch_size"],
            num_workers=0)     
        
        return data_loader
    
    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        bottleneck = self.encoder_block3(x)
        
        x = self.decoder_block1(bottleneck)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        return x, bottleneck

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        images = batch 
        image_yhat, bottleneck = self.forward(images) 
        
        #Calculate losses
        loss = F.mse_loss(image_yhat, images)    
        self.log("train_loss", loss, on_epoch=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        images = batch 
        image_yhat, bottleneck = self.forward(images) 
        
        #Calculate losses
        loss = F.mse_loss(image_yhat, images)    
        output = self.metrics(image_yhat, images) 
        self.log_dict(output)   
        self.log("val_loss", loss, on_epoch=True)
        
        return loss
        
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.0001)
        
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