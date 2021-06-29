#Lightning Module
from src import utils
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer

class TreeModel(LightningModule):
    def __init__(self,*args, **kwargs):
        super().__init__()
    
        #read config
        self.config = utils.read_config()
        
        #create model
    def create_trainer(self, comet_logger=None):
        """Create a trainer from the config file parameters"""
        self.trainer = Trainer(logger=comet_logger, gpus=self.config["gpus"], fast_dev_run=self.config["train"]["fast_dev_run"], accelerator=self.config["train"]["accelerator"])
        
        return self.trainer