#Lightning Module
from src import utils
from pytorch_lightning import LightningModule

class TreeModel(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
        #read config
        self.config = utils.read_config()
        
        #create model
        