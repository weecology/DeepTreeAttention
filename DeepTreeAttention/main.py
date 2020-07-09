#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics

#Local Modules
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.models import Hang2020, single_conv
from DeepTreeAttention.generators.make_dataset import tf_dataset
from DeepTreeAttention.callbacks import callbacks

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
            self.model = load_model(saved_model)

        #Holders
        self.testing_set = None
        self.training_set = None
    
    def get_model(self, name): 
        #Define shape
        classes = self.config["train"]["classes"]
        height = self.config["train"]["crop_size"]
        width = self.config["train"]["crop_size"]
        channels = self.config["train"]["sensor_channels"]
        
        if name == "Hang2020":
            weighted_sum=self.config["train"]["weighted_sum"]            
            return Hang2020.create_model(height, width, channels, classes, weighted_sum)
        
        elif name == "single_conv":
            return single_conv.create_model(height, width, channels, classes)
        else:
            raise ValueError("Unknown model name {}",format(name))

    def create(self, name="Hang2020",weights=None):
        """Load a model
            Args:
                weights: a saved model weights from previous run
                name: a model name from DeepTreeAttention.models
            """
        self.model = self.get_model(name)
        
        if weights:
            self.model.load_weights(weights)

        #metrics
        metric_list = [
            metrics.TopKCategoricalAccuracy(k=2, name="top_k"),
            metrics.CategoricalAccuracy(name="acc")]
        
        #compile
        self.model.compile(loss="categorical_crossentropy",
                           optimizer=tf.keras.optimizers.Adam(
                               lr=float(self.config['train']['learning_rate'])),
                           metrics=metric_list)

    def read_data(self, validation_split=False):
        """Read tfrecord into datasets from config
            Args:
                validation_split: True -> split tfrecords into train test. This overrides the evaluation config!
            """
        self.train_records = glob.glob(os.path.join(self.config["train"]["tfrecords"], "*.tfrecord"))
        
        if validation_split:
            print("Splitting training set into train-test")
            train_df = pd.Series(self.train_records)
            self.train_split_records = train_df.sample(frac=0.9).values
            self.test_split_records = train_df[~(train_df.isin(self.train_split_records))].values
            
            #Create training tf.data
            self.train_split = tf_dataset(
                tfrecords=self.train_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"]
            )
            #Create testing tf.data
            self.val_split = tf_dataset(
                tfrecords=  self.test_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"]
            )            
        else:
            #Create training tf.data
            self.train_split = tf_dataset(
                tfrecords=self.train_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"]
            )
            
            #honor config if validation not set
            self.val_split = None     
            if self.config["evaluation"]["tfrecords"] is not None:
                self.test_records = glob.glob(os.path.join(self.config["evaluation"]["tfrecords"], "*.tfrecord"))
                self.val_split = tf_dataset(
                    tfrecords = self.test_records, 
                    batch_size = self.config["train"]["batch_size"],
                    shuffle = self.config["train"]["shuffle"])

    def train(self):
        """Train a model"""       
        
        callback_list = callbacks.create()
        
        self.model.fit(
            self.train_split,
            epochs=self.config["train"]["epochs"],
            validation_data=self.val_split,
            callbacks=callback_list,
        )

    def predict(self, tfrecords, batch_size=1):
        """Predicted a set of tfrecords and create a raster image"""
        prediction_set = tf_dataset(
            tfrecords = tfrecords, 
            batch_size = batch_size,
            shuffle = False,
            train=False)
        
        predictions = []
        row_list = []
        col_list = []
        for image, x,y in prediction_set:
            try:
                softmax_batch = self.model.predict_on_batch(image)
                for row, col, i in zip(x.numpy(), y.numpy(), softmax_batch):
                    label = np.argmax(i)
                    predictions.append(label)
                    row_list.append(row)
                    col_list.append(col) 
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))

        results = pd.DataFrame({"label":predictions,"row":row_list,"col":col_list})
        results = results.sort_values(by=["row","col"])
        
        return results

    def evaluate(self, tf_dataset):
        """Evaluate metrics on held out training data. Defaults to reading from config.yml evaluation sensor path
        Args: 
            tf_dataset: Optional a tf.dataset that yields data and labels, see make_dataset.py 
            steps: Optional, how many calls of the genertor to evaluate. None will evaluate until exhausted
        Returns:
            results: a dictionary of metrics
        """
        #gather y_true
        labels = []
        predictions = []
        counter = 0
        for image, label in tf_dataset:
            try:
                counter+=1
                print(counter)
                softmax_batch = self.model.predict_on_batch(image)
                one_hot_label = label.numpy()
                predictions.append(softmax_batch)
                labels.append(label)
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))
        
        #Create numpy arrays of batches
        predictions = np.vstack(predictions)
        labels = np.vstack(labels)
        
        return predictions, labels
