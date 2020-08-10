#Wrapper class for DeepTreeAttention
"""Wrap generate data, create, train and predict into a single set of class commands"""
import os
import re
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
from DeepTreeAttention.generators import boxes
from DeepTreeAttention.callbacks import callbacks
from DeepTreeAttention.utils import Hyperspectral

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

        #log some helpful data
        self.classes = self.config["train"]["classes"]
        self.height = self.config["train"]["crop_size"]
        self.width = self.config["train"]["crop_size"]
        self.channels = self.config["train"]["sensor_channels"]
        self.weighted_sum = self.config["train"]["weighted_sum"]
    
    def find_hyperspectral_path(self, shapefile, lookup_pool):
        """Find a hyperspec path based on the shapefile using NEONs schema"""
        pool = glob.glob(lookup_pool, recursive=True)
        basename = os.path.splitext(os.path.basename(shapefile))[0]        
        
        #Get file metadata from name string
        geo_index = re.search("(\d+_\d+)_image",basename).group(1)
        year = re.search("(\d+)_",basename).group(1)
        
        match = [x for x in pool if geo_index in x]
        
        #of the matches get the correct year
        year_match = [x for x in match if year in x]
        
        if len(year_match) == 0:
            raise ValueError("No matching tile in {} for shapefile {}".format(lookup_pool, shapefile))
        elif len(year_match) > 1:
            raise ValueError("Multiple matching tiles in {} for shapefile {}".format(lookup_pool, shapefile))
        else:
            return year_match[0]
    
    def find_rgb_path(self, shapefile, lookup_pool):
        """Find a hyperspec path based on the shapefile using NEONs schema"""
        pool = glob.glob(lookup_pool, recursive=True)
        basename = os.path.splitext(os.path.basename(shapefile))[0]        
        
        #Get file metadata from name string
        geo_index = re.search("(\d+_\d+)_image",basename).group(1)
        year = re.search("(\d+)_",basename).group(1)
        
        match = [x for x in pool if geo_index in x]
        
        #of the matches get the correct year
        year_match = [x for x in match if year in x]
        
        if len(year_match) == 0:
            raise ValueError("No matching rgb tile in {} for shapefile {}".format(lookup_pool, shapefile))
        elif len(year_match) > 1:
            raise ValueError("Multiple matching rgb tiles in {} for shapefile {}".format(lookup_pool, shapefile))
        else:
            return year_match[0]
        
    def generate(self, shapefile, train=True, sensor_path=None, chunk_size=1000):
        """Predict species class for each DeepForest bounding box
        Args:
            shapefile: a DeepForest shapefile (see NeonCrownMaps) with a bounding box and utm projection
            train: generate a training record that yields, image, label, or a prediction record with metadata? Default True
            sensor_path: supply a known path to a sensor geoTIFF tile. If not, use a lookup function hardcoded to a dir
            chunk_size: number of crops per tfrecord
        """
        if sensor_path is None:
            if train:
                hyperspectral_h5_path = self.find_hyperspectral_path(shapefile, lookup_pool=self.config["train"]["hyperspectral_sensor_pool"])
                rgb_path = self.find_rgb_path(shapefile, lookup_pool=self.config["train"]["rgb_sensor_pool"])                
            else:
                hyperspectral_h5_path = self.find_hyperspectral_path(shapefile, lookup_pool=self.config["train"]["hyperspectral_sensor_pool"])
                rgb_path = self.find_rgb_path(shapefile, lookup_pool=self.config["train"]["rgb_sensor_pool"])                
        
        #convert .h5 hyperspec tile if needed
        if os.path.splitext[1] == ".h5":
            tif_basename = os.path.splitext(os.path.basename(rgb_path))[0] + "_hyperspectral.tif"    
            tif_path = "{}/{}".format(self.config["hyperspectral_tif_dir"], tif_basename)
            
            if not os.path.exists(tif_path):
                sensor_path = Hyperspectral.generate_raster(h5_path = hyperspectral_h5_path, rgb_filename=rgb_path, bands="All", save_dir=self.config["hyperspectral_tif_dir"])
            else:
                sensor_path = tif_path
        
        #set savedir
        if train:
            savedir = self.config["train"]["tfrecords"]
        else:
            savedir = self.config["predict"]["tfrecords"]
            
        created_records = boxes.generate_tfrecords(
            shapefile=shapefile,
            sensor_path=sensor_path,
            height=self.height,
            width=self.width,
            savedir=savedir,
            train=train,
            classes=self.classes,
            chunk_size=chunk_size
        )
    
        return created_records
        
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
                
        if self.config["train"]["gpu"] > 1:
            strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    
            #Store intermediary layers for subtraining
            with strategy.scope():
                #metrics
                metric_list = [
                    metrics.CategoricalAccuracy(name="acc")
                ]
                
                #Create model
                self.inputs, self.combined_output, self.spatial_attention_outputs, self.spectral_attention_outputs = Hang2020.create_model(
                    self.height, self.width, self.channels, self.classes, self.weighted_sum)
        
                #Full model compile
                self.model = tf.keras.Model(inputs=self.inputs,
                                            outputs=self.combined_output,
                                            name="DeepTreeAttention")
                
                #compile full model
                self.model.compile(loss="categorical_crossentropy",
                                           optimizer=tf.keras.optimizers.Adam(
                                               lr=float(0.0001)),
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
        else:
            #metrics
            metric_list = [
                metrics.CategoricalAccuracy(name="acc")
            ]
            
            #Create model
            self.inputs, self.combined_output, self.spatial_attention_outputs, self.spectral_attention_outputs = Hang2020.create_model(
                self.height, self.width, self.channels, self.classes, self.weighted_sum)
    
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
            self.train_split_records = train_df.head(int(self.config["train"]["training_fraction"] * train_df.shape[0])).values
            self.test_split_records = train_df[~(
                train_df.isin(self.train_split_records))].values

            #Create training tf.data
            self.train_split = boxes.tf_dataset(tfrecords=self.train_split_records,
                                          batch_size=self.config["train"]["batch_size"],
                                          shuffle=self.config["train"]["shuffle"],
                                          mode=mode,
                                          cores=self.config["cpu_workers"])
            
            #Create testing tf.data
            self.val_split = boxes.tf_dataset(tfrecords=self.test_split_records,
                                        batch_size=self.config["train"]["batch_size"],
                                        shuffle=self.config["train"]["shuffle"],
                                        mode=mode,
                                        cores=self.config["cpu_workers"])
        else:
            #Create training tf.data
            self.train_split = boxes.tf_dataset(tfrecords=self.train_records,
                                          batch_size=self.config["train"]["batch_size"],
                                          shuffle=self.config["train"]["shuffle"],
                                          mode=mode,
                                        cores=self.config["cpu_workers"])

            #honor config if validation not set
            self.val_split = None
            if self.config["evaluation"]["tfrecords"] is not None:
                self.test_records = glob.glob(os.path.join(self.config["evaluation"]["tfrecords"], "*.tfrecord"))
                
                self.val_split = boxes.tf_dataset(tfrecords=self.test_records,
                                            batch_size=self.config["train"]["batch_size"],
                                            shuffle=self.config["train"]["shuffle"],
                                            mode=mode,
                                            cores=self.config["cpu_workers"])

    def train(self, class_weight=None, submodel=None):
        """Train a model"""
        callback_list = callbacks.create(self.log_dir)
        
        if submodel == "spatial":
            #The spatial model is very shallow compared to spectral, train for longer
            self.spatial_model.fit(self.train_split,
                               epochs=int(self.config["train"]["epochs"]/2),
                               validation_data=self.val_split,
                               callbacks=callback_list,
                               class_weight=class_weight)
            
        elif submodel == "spectral":
            #one for each loss layer
            self.spectral_model.fit(self.train_split,
                               epochs=int(self.config["train"]["epochs"]/2),
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
        prediction_set = boxes.tf_dataset(tfrecords=tfrecords,
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
    
    def predict(self, shapefile, savedir, create_records=True, sensor_path=None):
        """Predict species id for each box in a single shapefile
        Args:
            shapefile: path to a shapefile
            record_dirname: directory to save generated records
            create_records: overwrite previous records
        Returns:
            fname: path to predicted shapefile
        """                
        if create_records:
            created_records = boxes.generate(shapefile, sensor_path=sensor_path, savedir = self.config["predict"]["savedir"], height=self.height, width=self.width, classes=self.classes, train=False)
        else:
            created_records = glob.glob(dirname + "*.tfrecord")
        
        #Merge with original box shapefile by index and write new shapefile to file
        results = self.predict_boxes(created_records)
        fname = self.merge_shapefile(shapefile, results, savedir=savedir)
        
        return fname
    
    def predict_boxes(self, tfrecords, batch_size=1):
        """Predicted a set of tfrecords and create a raster image"""
        prediction_set = boxes.tf_dataset(tfrecords=tfrecords,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    mode="predict",
                                    cores=self.config["cpu_workers"])

        predictions = []
        indices = []
        for image, box_index in prediction_set:
            try:
                softmax_batch = self.model.predict_on_batch(image)
                predictions.append(softmax_batch)
                indices.append(box_index)
            except tf.errors.OutOfRangeError:
                print("Completed {} predictions".format(len(predictions)))

        #stack
        predictions = np.vstack(predictions)
        predictions = np.argmax(predictions, 1)
        
        indices = np.concatenate(indices)
        labels = [self.config["class_labels"][x] for x in predictions]
        results = pd.DataFrame({"label": labels, "box_index": indices})
        
        #decode results
        results["box_index"] = results["box_index"].apply(lambda x: x.decode()).astype(str)
        
        return results
    
    def merge_shapefile(self, shapefile, results, savedir):
        """Merge predicted species label with box id"""
        
        gdf = geopandas.read_file(shapefile)
        
        #Make sure there isn't a label column in merge data
        gdf = gdf.drop(columns="label")
        basename = os.path.splitext(os.path.basename(shapefile))[0]
        gdf["box_index"] = ["{}_{}".format(basename, x) for x in gdf.index.values]
        
        #Merge 
        joined_gdf = gdf.merge(results, on="box_index")
        fname = "{}/{}.shp".format(savedir, basename)
        joined_gdf.to_file(fname)
        
        return fname
    
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
