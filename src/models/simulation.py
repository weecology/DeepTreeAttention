#Autoencoder
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch import optim
from src.models.Hang2020 import conv_module
from src import visualize
from src import data
import numpy as np
from pytorch_lightning import LightningModule
import pandas as pd
import torchmetrics
from matplotlib import pyplot as plt

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
    def __init__(self, bands, classes, config):
        super(autoencoder, self).__init__()    
        
        self.config = config
        
        #Encoder
        self.encoder_block1 = encoder_block(in_channels=bands, filters=64, pool=True)
        self.encoder_block2 = encoder_block(in_channels=64, filters=32, pool=True)
        self.encoder_block3 = encoder_block(in_channels=32, filters=16, pool=True)
        
        #Classification layer
        self.vis_layer = nn.Linear(in_features=12544, out_features=2)
        self.fc1 = nn.Linear(in_features=2, out_features=classes)
        
        #Decoder
        self.decoder_block1 = decoder_block(in_channels=16, filters=32)
        self.decoder_block2 = decoder_block(in_channels=32, filters=64)
        self.decoder_block3 = decoder_block(in_channels=64, filters=bands)
        
        #Visualization
        # a dict to store the activations        
        self.vis_activation = {}
        def getActivation(name):
            # the hook signature
            def hook(model, input, output):
                self.vis_activation[name] = output.detach()
            return hook
        
        self.vis_layer.register_forward_hook(getActivation("vis_layer"))        
        self.encoder_block3.register_forward_hook(getActivation("encoder_block3"))        

        #Metrics
        micro_recall = torchmetrics.Accuracy(average="micro")
        self.metrics = torchmetrics.MetricCollection({"Micro Accuracy":micro_recall}, prefix="autoencoder")

    def forward(self, x):
        x = self.encoder_block1(x)
        x = self.encoder_block2(x)
        x = self.encoder_block3(x)
        
        #vis layer projection
        y = x.view(-1, 16*28*28)
        y = self.vis_layer(y)
        y = F.relu(y)
        y = self.fc1(y)
        y = F.relu(y)

        x = self.decoder_block1(x)
        x = self.decoder_block2(x)
        x = self.decoder_block3(x)

        return x, y

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        images, observed_labels, true_labels = batch 
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)    
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        loss = autoencoder_loss + (classification_loss * 0.1)
        
        softmax_prob = F.softmax(classification_yhat, dim =1)
        output = self.metrics(softmax_prob, true_labels) 
        self.log_dict(output, on_epoch=True, on_step=False)
        
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])

        return {'optimizer':optimizer,"monitor":'val_loss'}

    def on_train_end(self):
        """At the end of each epoch trigger the dataset to collect intermediate activation for plotting"""
        #plot 2d projection layer
        epoch_labels = []
        vis_epoch_activations = []
        encoder_epoch_activations = []
        
        for batch in self.train_dataloader():
            images, observed_labels, true_labels  = batch
            epoch_labels.append(observed_labels)
            #trigger activation hook
            if next(self.parameters()).is_cuda:
                image = images.cuda()
            else:
                image = images
            
            pred = self(image)
            vis_epoch_activations.append(self.vis_activation["vis_layer"].cpu())
            encoder_epoch_activations.append(self.vis_activation["encoder_block3"].cpu())

        #Create a single array
        epoch_labels = np.concatenate(epoch_labels)
        vis_epoch_activations = torch.tensor(np.concatenate(vis_epoch_activations))
        encoder_epoch_activations = torch.tensor(np.concatenate(encoder_epoch_activations))
        
        layerplot_vis = visualize.plot_2d_layer(vis_epoch_activations, epoch_labels)
        try:
            self.logger.experiment.log_figure(figure=layerplot_vis, figure_name="2d_vis_projection", step=self.current_epoch)
        except Exception as e:
            print("Comet logger failed: {}".format(e))
            
        layerplot_encoder = visualize.plot_2d_layer(encoder_epoch_activations, epoch_labels, use_pca=True)
        try:
            self.logger.experiment.log_figure(figure=layerplot_encoder, figure_name="2d_encoder_projection", step=self.current_epoch)
        except Exception as e:
            print("Comet logger failed: {}".format(e))
                
        #reset activations
        self.vis_epoch_activations = {}
        self.encoder_epoch_activations = {}
        
        
        
    
            
