#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
from models import create_model
from generators.make_dataset import training_dataset
from tensorflow.keras.models import load_model

class AttentionModel():
    """The main class holding train, predict and evaluate methods"""
    def __init__(self, config="../conf/config.yml", saved_model=None):
        """
        Args:
            config: path to a config file, defaults to ../conf/config.yml
            saved_model: Optional, a previous saved AttentionModel .h5
        """
        self.config = utils.parse_yaml()
        if saved_model:
            load_model(saved_model)
            
    def create_model(self, classes, weights=None):
        """weights: a saved model weights from previous run"""
        #Infer classes 
        self.classses = classes
        self.model = create_model(self.classes)
        if weights:
            self.model.load_weights(weights)
            
    def train(self):
        
    
    def predict(self):
        pass
    
    def evaluate(self):
        pass
    
