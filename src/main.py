#Lightning Data Module
import os
from pytorch_lightning import LightningModule
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
        
