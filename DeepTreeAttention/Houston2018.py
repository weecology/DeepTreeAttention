#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import glob
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import metrics
from tensorflow.keras.utils import multi_gpu_model

from sklearn.utils import class_weight

#Local Modules
from DeepTreeAttention.utils.config import parse_yaml
from DeepTreeAttention.models import Hang2020
from DeepTreeAttention.generators import make_dataset
from DeepTreeAttention.callbacks import callbacks


class AttentionModel():
    """The main class holding train, predict and evaluate methods"""

    def __init__(self, config="conf/config.yml", saved_model=None, log_dir=None):
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

        if log_dir:
            self.log_dir = log_dir
        else:
            self.log_dir = None

    def calc_class_weight(self):
        """Get class frequency of labels"""

        #Check if train_split has been create
        if not hasattr(self, "train_split"):
            raise ValueError(
                "No training split created, please call DeepTreeAttention.read_data()")

        labels = []
        for image, label in self.train_split:
            labels.append(label)

        #Convert from one_hot
        labels = np.vstack(labels)
        labels = np.argmax(labels, 1)

        class_weights = class_weight.compute_class_weight('balanced', np.unique(labels),
                                                          labels)

        return class_weights

    def create(self, weights=None, submodel=None):
        """Load a model
            Args:
                weights: a saved model weights from previous run
                name: a model name from DeepTreeAttention.models
            """
        #Tensorflow suggest create on CPU to reduce memory
        #If more than GPU is requested

        classes = self.config["train"]["classes"]
        height = self.config["train"]["crop_size"]
        width = self.config["train"]["crop_size"]
        channels = self.config["train"]["sensor_channels"]
        weighted_sum = self.config["train"]["weighted_sum"]

        if self.config["train"]["gpu"] > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))

            #Store intermediary layers for subtraining
            with strategy.scope():
                #metrics
                metric_list = [metrics.CategoricalAccuracy(name="acc")]

                #Create model
                self.inputs, self.combined_output, self.spatial_attention_outputs, self.spectral_attention_outputs = Hang2020.create_model(
                    height, width, channels, classes, weighted_sum)

                #Full model compile
                self.model = tf.keras.Model(inputs=self.inputs,
                                            outputs=self.combined_output,
                                            name="DeepTreeAttention")

                #compile full model
                self.model.compile(loss="categorical_crossentropy",
                                   optimizer=tf.keras.optimizers.Adam(lr=float(0.0001)),
                                   metrics=metric_list)
                #compile
                loss_dict = {
                    "spatial_attention_1": "categorical_crossentropy",
                    "spatial_attention_2": "categorical_crossentropy",
                    "spatial_attention_3": "categorical_crossentropy"
                }

                # Spatial Attention softmax model
                self.spatial_model = tf.keras.Model(
                    inputs=self.inputs,
                    outputs=self.spatial_attention_outputs,
                    name="DeepTreeAttention")

                self.spatial_model.compile(
                    loss=loss_dict,
                    loss_weights=[0.01, 0.1, 1],
                    optimizer=tf.keras.optimizers.Adam(
                        lr=float(self.config['train']['learning_rate'])),
                    metrics=metric_list)

                # Spectral Attention softmax model
                self.spectral_model = tf.keras.Model(
                    inputs=self.inputs,
                    outputs=self.spectral_attention_outputs,
                    name="DeepTreeAttention")

                #compile loss dict
                loss_dict = {
                    "spectral_attention_1": "categorical_crossentropy",
                    "spectral_attention_2": "categorical_crossentropy",
                    "spectral_attention_3": "categorical_crossentropy"
                }

                self.spectral_model.compile(
                    loss=loss_dict,
                    loss_weights=[0.01, 0.1, 1],
                    optimizer=tf.keras.optimizers.Adam(
                        lr=float(self.config['train']['learning_rate'])),
                    metrics=metric_list)

                if weights:
                    self.model.load_weights(weights)
        else:
            #metrics
            metric_list = [metrics.CategoricalAccuracy(name="acc")]

            #Create model
            self.inputs, self.combined_output, self.spatial_attention_outputs, self.spectral_attention_outputs = Hang2020.create_model(
                height, width, channels, classes, weighted_sum)

            #Full model compile
            self.model = tf.keras.Model(inputs=self.inputs,
                                        outputs=self.combined_output,
                                        name="DeepTreeAttention")

            #compile full model
            self.model.compile(loss="categorical_crossentropy",
                               optimizer=tf.keras.optimizers.Adam(
                                   lr=float(self.config['train']['learning_rate'])),
                               metrics=metric_list)
            #compile
            loss_dict = {
                "spatial_attention_1": "categorical_crossentropy",
                "spatial_attention_2": "categorical_crossentropy",
                "spatial_attention_3": "categorical_crossentropy"
            }

            # Spatial Attention softmax model
            self.spatial_model = tf.keras.Model(inputs=self.inputs,
                                                outputs=self.spatial_attention_outputs,
                                                name="DeepTreeAttention")

            self.spatial_model.compile(
                loss=loss_dict,
                loss_weights=[0.01, 0.1, 1],
                optimizer=tf.keras.optimizers.Adam(
                    lr=float(self.config['train']['learning_rate'])),
                metrics=metric_list)

            # Spectral Attention softmax model
            self.spectral_model = tf.keras.Model(inputs=self.inputs,
                                                 outputs=self.spectral_attention_outputs,
                                                 name="DeepTreeAttention")

            #compile loss dict
            loss_dict = {
                "spectral_attention_1": "categorical_crossentropy",
                "spectral_attention_2": "categorical_crossentropy",
                "spectral_attention_3": "categorical_crossentropy"
            }

            self.spectral_model.compile(
                loss=loss_dict,
                loss_weights=[0.01, 0.1, 1],
                optimizer=tf.keras.optimizers.Adam(
                    lr=float(self.config['train']['learning_rate'])),
                metrics=metric_list)

            if weights:
                self.model.load_weights(weights)

    def read_data(self, mode="train", validation_split=False):
        """Read tfrecord into datasets from config
            Args:
                validation_split: True -> split tfrecords into train test. This overrides the evaluation config!
            """
        self.train_records = glob.glob(
            os.path.join(self.config["train"]["tfrecords"], "*.tfrecord"))

        if validation_split:
            print("Splitting training set into train-test")
            train_df = pd.Series(self.train_records)
            #Sample with set seed to make it the same between runs
            self.train_split_records = train_df.head(int(0.9 * train_df.shape[0])).values
            self.test_split_records = train_df[~(
                train_df.isin(self.train_split_records))].values

            #Create training tf.data
            self.train_split = make_dataset.tf_dataset(
                tfrecords=self.train_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])

            #Create testing tf.data
            self.val_split = make_dataset.tf_dataset(
                tfrecords=self.test_split_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])
        else:
            #Create training tf.data
            self.train_split = make_dataset.tf_dataset(
                tfrecords=self.train_records,
                batch_size=self.config["train"]["batch_size"],
                shuffle=self.config["train"]["shuffle"],
                mode=mode,
                cores=self.config["cpu_workers"])

            #honor config if validation not set
            self.val_split = None
            if self.config["evaluation"]["tfrecords"] is not None:
                self.test_records = glob.glob(
                    os.path.join(self.config["evaluation"]["tfrecords"], "*.tfrecord"))

                self.val_split = make_dataset.tf_dataset(
                    tfrecords=self.test_records,
                    batch_size=self.config["train"]["batch_size"],
                    shuffle=self.config["train"]["shuffle"],
                    mode=mode,
                    cores=self.config["cpu_workers"])

    def train(self, class_weight=None, submodel=None):
        """Train a model"""

        callback_list = callbacks.create(self.log_dir, validation_data=self.val_split)

        if submodel == "spatial":
            #The spatial model is very shallow compared to spectral, train for longer
            self.spatial_model.fit(self.train_split,
                                   epochs=int(self.config["train"]["epochs"] / 2),
                                   validation_data=self.val_split,
                                   callbacks=callback_list,
                                   class_weight=class_weight)

        elif submodel == "spectral":
            #one for each loss layer
            self.spectral_model.fit(self.train_split,
                                    epochs=int(self.config["train"]["epochs"] / 2),
                                    validation_data=self.val_split,
                                    callbacks=callback_list,
                                    class_weight=class_weight)
        else:
            self.model.fit(self.train_split,
                           epochs=self.config["train"]["epochs"],
                           validation_data=self.val_split,
                           callbacks=callback_list,
                           class_weight=class_weight)

    def predict_raster(self, tfrecords, batch_size=1):
        """Predicted a set of tfrecords and create a raster image"""
        prediction_set = make_dataset.tf_dataset(tfrecords=tfrecords,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 mode="predict",
                                                 cores=self.config["cpu_workers"])

        predictions = []
        row_list = []
        col_list = []
        for image, x, y in prediction_set:
            try:
                softmax_batch = self.model.predict_on_batch(image)
                row_list.append(x.numpy())
                col_list.append(y.numpy())
                predictions.append(softmax_batch)
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))

        #stack
        predictions = np.vstack(predictions)
        row_list = np.concatenate(row_list)
        col_list = np.concatenate(col_list)
        predictions = np.argmax(predictions, 1)
        results = pd.DataFrame({"label": predictions, "row": row_list, "col": col_list})
        results = results.sort_values(by=["row", "col"])

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
        for image, label in tf_dataset:
            try:
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
