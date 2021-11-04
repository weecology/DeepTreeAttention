#Autoencoder
from torch.nn import functional as F
import torch
import torch.nn as nn
from torch import optim
from src.models.Hang2020 import conv_module
from pytorch_lightning import LightningModule
import torchmetrics

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
        self.vis_conv1= encoder_block(in_channels=16, filters=8)        
        self.vis_layer = nn.Linear(in_features=6272, out_features=2)
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
        
        #classification layer projection
        y = self.vis_conv1(x)
        y = y.view(-1, 8*28*28)        
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
        index, images, observed_labels, true_labels = batch 
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)    
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        #loss = autoencoder_loss + (classification_loss * 0.1)

        return classification_loss

    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        index, images, observed_labels, true_labels = batch 
        autoencoder_yhat, classification_yhat = self.forward(images) 
        
        autoencoder_loss = F.mse_loss(autoencoder_yhat, images)    
        classification_loss = F.cross_entropy(classification_yhat, observed_labels)
        #loss = autoencoder_loss + (classification_loss * 0.1)
        
        softmax_prob = F.softmax(classification_yhat, dim =1)
        output = self.metrics(softmax_prob, true_labels) 
        self.log("val_loss", classification_loss, on_epoch=True)
        self.log_dict(output, on_epoch=True, on_step=False)
        
        return classification_loss
            
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["lr"])

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
        
        
        
    
            
