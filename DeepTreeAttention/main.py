#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

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

        #Holders
        self.testing_set = None
        self.training_set = None

    def create(self, weights=None):
        """weights: a saved model weights from previous run"""
        self.model = create_model.model(classes=self.config["train"]["classes"],
                                        height=self.config["train"]["crop_size"],
                                        channels=self.config["train"]["sensor_channels"],
                                        weighted_sum=self.config["train"]["weighted_sum"])

        if weights:
            self.model.load_weights(weights)

        #metrics
        metric_list = [
            metrics.TopKCategoricalAccuracy(k=5, name="top_k"),
            metrics.Accuracy(name="acc")]
        
        #compile
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=tf.keras.optimizers.Adam(
                               lr=float(self.config['train']['learning_rate'])),
                           metrics=metric_list)

    def read_data(self):
        
        #Create training tf.data
        self.training_set = tf_dataset(
        sensor_path=self.config["train"]["sensor_path"],
        ground_truth_path=self.config["train"]["ground_truth_path"],
        size=self.config['train']["crop_size"],
        batch_size=self.config["train"]["batch_size"],
        classes=self.config["train"]["classes"],
        repeat=False,
        shuffle=self.config["train"]["shuffle"])

        if self.config["evaluation"]["sensor_path"] is not None:
            self.testing_set = tf_dataset(
            sensor_path = self.config["evaluation"]["sensor_path"],
            ground_truth_path = self.config["evaluation"]["ground_truth_path"],
            size = self.config['train']["crop_size"],
            batch_size = self.config["train"]["batch_size"],
            classes = self.config["train"]["classes"],
            repeat = False,
            shuffle = self.config["train"]["shuffle"])
            
        else:
            self.testing_set = None
            
    def train(self):
        """Train a model"""
        
        self.model.fit(self.training_set,
                       epochs=self.config["train"]["epochs"],
                       steps_per_epoch=self.config["train"]["steps"])

    def predict(self):
        pass

    def evaluate(self, tf_dataset=None, steps=None):
        """Evaluate metrics on testing data. Defaults to reading from config.yml evaluation sensor path
        Args: 
            tf_dataset: Optional a tf.dataset that yields data and labels, see make_dataset.py 
            steps: Optional, how many calls of the genertor to evaluate. None will evaluate until exhausted
        Returns:
            results: a dictionary of metrics
        """
        if tf_dataset:
            result = self.model.evaluate(x=tf_dataset, steps=steps)
        else:
            if self.testing_set is None:
                raise IOError(
                    "Testing set is not specified and config.yml has no evaluation: sensor path set"
                )

            result = self.model.evaluate(self.testing_set, steps=steps)
            
        result = dict(zip(self.model.metrics_names, result))            
        
        return result
