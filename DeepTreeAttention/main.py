#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
#Local Modules
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.models import create_model
from DeepTreeAttention.generators.make_dataset import tf_dataset

class AttentionModel():
    """The main class holding train, predict and evaluate methods"""
    def __init__(self, config="conf/config.yml", saved_model=None):
        """
        Args:
            config: path to a config file, defaults to ../conf/config.yml
            saved_model: Optional, a previous saved AttentionModel .h5
        """
        self.config = parse_yaml(config)
        if saved_model:
            load_model(saved_model)
            
    def create(self, classes, weights=None, weighted_sum=True):
        """weights: a saved model weights from previous run"""
        #Infer classes 
        self.classes = classes
        self.model = create_model.model(
            classes=self.classes,
            height=self.config["train"]["crop_height"],
            width=self.config["train"]["crop_width"],
            channels=self.config["train"]["sensor_channels"],
            weighted_sum = weighted_sum)
            
        if weights:
            self.model.load_weights(weights)
            
    def train(self):
        
        #Create training tf.data
        training_set = tf_dataset(
            sensor_path = self.config["train"]["sensor_path"],
            ground_truth_path = self.config["train"]["ground_truth_path"],
            crop_height = self.config['train']["crop_height"],
            crop_width = self.config['train']["crop_width"],            
            sensor_channels = self.config["train"]["sensor_channels"],
            batch_size = self.config["train"]["batch_size"]
        )
        
        if self.config["evaluation"]["sensor_path"] is not None:
            training_set = tf_dataset(
                sensor_path = self.config["evaluation"]["sensor_path"],
                ground_truth_path = self.config["evaluation"]["ground_truth_path"],
                crop_height = self.config['train']["crop_height"],
                crop_width = self.config['train']["crop_width"],            
                sensor_channels = self.config["train"]["sensor_channels"],
                batch_size = self.config["train"]["batch_size"]
            ) 
        else:
            testing_set = None
        
        #compile
        self.model.compile(
            loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(lr=float(self.config['train']['learning_rate']))
        )
        
        self.model.fit(
            training_set,
            epochs=self.config["train"]["epochs"],
            steps_per_epoch=self.config["train"]["steps"])
    
    def predict(self):
        pass
    
    def evaluate(self):
        result = self.model.evaluate(testing_set)
        return dict(zip(self.model.metrics_names, result))
    
