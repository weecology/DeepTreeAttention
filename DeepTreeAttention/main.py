#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
from models import create_model

class AttentionModel():
    
    def __init__(self, config):
        """classes: number of classes in model"""
        self.config = utils.parse_yaml()
    
    def create_model(self, weights=None):
        """weights: a saved model weights from previous run """
        #Infer classes 
        self.model = create_model(self.classes)
        if weights:
            self.model.load_weights(weights)
    
    def train(self):
        self.model.fit()
      
    
