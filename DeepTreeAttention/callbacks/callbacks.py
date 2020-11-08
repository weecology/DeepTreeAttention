#Callbacks
"""Create training callbacks"""

import os
import numpy as np
import pandas as pd

from datetime import datetime
from DeepTreeAttention.utils import metrics
from DeepTreeAttention.visualization import visualize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow import expand_dims

class F1Callback(Callback):

    def __init__(self, experiment, eval_dataset, y_true, label_names, submodel, n=6):
        """F1 callback
        Args:
            n: number of epochs to run. If n=4, function will run every 4 epochs
            y_true: instead of iterating through the dataset every time, just do it once and pass the true labels to the function
        """
        self.experiment = experiment
        self.eval_dataset = eval_dataset
        self.label_names = label_names
        self.submodel = submodel
        self.n = n
        self.y_true = y_true
 
    def on_train_end(self, logs={}):
            
        y_pred = []
        sites = []
        
        #gather site and species matrix
        y_pred = self.model.predict(self.eval_dataset)
        
        if self.submodel in ["spectral","spatial"]:
            y_pred = y_pred[0]
        
        #F1
        macro, micro = metrics.f1_scores(self.y_true, y_pred)
        self.experiment.log_metric("Final MicroF1", micro)
        self.experiment.log_metric("Final MacroF1", macro)
        
        #Log number of predictions to make sure its constant
        self.experiment.log_metric("Prediction samples",y_pred.shape[0])
        results = pd.DataFrame({"true":np.argmax(self.y_true, 1),"predicted":np.argmax(y_pred, 1)})
        self.experiment.log_table("results_final.csv",results.values)
        
    def on_epoch_end(self, epoch, logs={}):
        
        if not epoch % self.n == 0:
            return None
            
        y_pred = []
        sites = []
        
        #gather site and species matrix
        y_pred = self.model.predict(self.eval_dataset)
        
        if self.submodel in ["spectral","spatial"]:
            y_pred = y_pred[0]
        
        #F1
        macro, micro = metrics.f1_scores(self.y_true, y_pred)
        self.experiment.log_metric("MicroF1", micro)
        self.experiment.log_metric("MacroF1", macro)
        
        #Log number of predictions to make sure its constant
        self.experiment.log_metric("Prediction samples",y_pred.shape[0])
        results = pd.DataFrame({"true":np.argmax(self.y_true, 1),"predicted":np.argmax(y_pred, 1)})
        self.experiment.log_table("results_{}.csv".format(epoch),results.values)
                               
class ConfusionMatrixCallback(Callback):

    def __init__(self, experiment, dataset, label_names, y_true, submodel):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel
        self.y_true = y_true
        
    def on_train_end(self, epoch, logs={}):
        
        y_pred = self.model.predict(self.dataset)
                    
        if self.submodel is "metadata":
            name = "Metadata Confusion Matrix"        
        elif self.submodel in ["ensemble"]:
            name = "Ensemble Matrix"
        else:
            name = "Confusion Matrix"

        cm = self.experiment.log_confusion_matrix(
            self.y_true,
            y_pred,
            title=name,
            file_name= name,
            labels=self.label_names,
            max_categories=80,
            max_example_per_cell=1)
        
        
class ImageCallback(Callback):

    def __init__(self, experiment, dataset, label_names, submodel=False):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel

    def on_train_end(self, epoch, logs={}):
        """Plot sample images with labels annotated"""

        #fill until there is atleast 20 images
        images = []
        y_pred = []
        y_true = []
        
        limit = 20
        num_images = 0
        for data, label in self.dataset:
            if num_images < limit:
                pred = self.model.predict(data)
                images.append(data)
                
                if self.submodel:
                    y_pred.append(pred[0])
                    y_true.append(label[0])
                else:
                    y_pred.append(pred)
                    y_true.append(label)                    

                num_images += label.shape[0]
            else:
                break
        
        images = np.vstack(images)
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)

        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)

        true_taxonID = [self.label_names[x] for x in y_true]
        pred_taxonID = [self.label_names[x] for x in y_pred]

        counter = 0
        for label, prediction, image in zip(true_taxonID, pred_taxonID, images):
            figure = visualize.plot_prediction(image=image,
                                               prediction=prediction,
                                               label=label)
            self.experiment.log_figure(figure_name="{}_{}".format(label, counter))
            counter += 1


def create(experiment, train_data, validation_data, log_dir=None, label_names=None, submodel=False):
    
    #turn off callbacks for metadata
    callback_list = []
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5,
                                  patience=10,
                                  min_delta=0.1,
                                  verbose=1)
    callback_list.append(reduce_lr)

    #Get the true labels since they are not shuffled
    y_true = [ ]
    for data, label in validation_data:
        if submodel in ["spatial","spectral"]:
            label = label[0]
        y_true.append(label)
            
    y_true = np.concatenate(y_true)
    
    if not submodel in ["spatial","spectral"]:
        confusion_matrix = ConfusionMatrixCallback(experiment=experiment, y_true=y_true, dataset=validation_data, label_names=label_names, submodel=submodel)
        callback_list.append(confusion_matrix)

    f1 = F1Callback(experiment=experiment, y_true=y_true, eval_dataset=validation_data, label_names=label_names, submodel=submodel)
    callback_list.append(f1)
    
    #if submodel is None:
        #plot_images = ImageCallback(experiment, validation_data, label_names, submodel=submodel)
        #callback_list.append(plot_images)
        
    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=10)

    return callback_list
