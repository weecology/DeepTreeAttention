#Lightning Data Module
import os
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch import optim

from src import data
from . import __file__

class TreeModel(LightningModule):
    """A pytorch lightning data module
    Args:
        model (str): Model to use. See the models/ directory. The name is the filename, each model should take in the same data loader
    """
    def __init__(self,model,config=None, *args, **kwargs):
        super().__init__()
    
        self.ROOT = os.path.dirname(os.path.dirname(__file__))    
        if config is None:
            self.config = data.read_config("{}/config.yml".format(self.ROOT))   
        else:
            self.config = config
        
        #Create model 
        self.model = model(bands = config["bands"], classes=config["classes"])
        
    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        images, y = batch
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)        
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        #allow for empty data if data augmentation is generated
        images, y = batch
        y_hat = self.model.forward(images)
        loss = F.cross_entropy(y_hat, y)        
        
        # Log loss
        self.log("val_loss", loss, on_epoch=True)
            
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                                   lr=self.config["train"]["lr"],
                                   momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode='min',
                                                         factor=0.1,
                                                         patience=10,
                                                         verbose=True,
                                                         threshold=0.0001,
                                                         threshold_mode='rel',
                                                         cooldown=0,
                                                         min_lr=0,
                                                         eps=1e-08)
        
        return {'optimizer':optimizer, 'lr_scheduler': scheduler,"monitor":'val_loss'}

    