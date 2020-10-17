#Callbacks
"""Create training callbacks"""

import os
import numpy as np

from datetime import datetime
from DeepTreeAttention.utils import metrics
from DeepTreeAttention.visualization import visualize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, TensorBoard
from tensorflow import expand_dims

class F1Callback(Callback):

    def __init__(self, experiment, train_dataset, eval_dataset, label_names, submodel, n=6):
        """F1 callback
        Args:
            n: number of epochs to run. If n=4, function will run every 4 epochs
        """
        self.experiment = experiment
        self.eval_dataset = eval_dataset
        self.train_dataset = train_dataset
        self.label_names = label_names
        self.submodel = submodel
        self.n = n

    def on_epoch_end(self, epoch, logs={}):
        
        if not epoch % self.n == 0:
            return None
            
        y_true = []
        y_pred = []
        sites = []
        
        #gather site and species matrix
        for data, label in self.eval_dataset:
            pred = self.model.predict(data)
            if self.submodel in ["spectral","spatial"]:
                y_pred.append(pred[0])
                y_true.append(label[0])
            else:
                y_pred.append(pred)
                y_true.append(label)       
        
        y_true_list = np.concatenate(y_true)
        y_pred_list = np.concatenate(y_pred)
        
        #F1
        macro, micro = metrics.f1_scores(y_true_list, y_pred_list)
        self.experiment.log_metric("MicroF1", micro)
        self.experiment.log_metric("MacroF1", macro)
        
class ConfusionMatrixCallback(Callback):

    def __init__(self, experiment, dataset, label_names, submodel):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel
        
    def on_train_end(self, epoch, logs={}):
        y_true = []
        y_pred = []

        for data, label in self.dataset:
            pred = self.model.predict(data)
            
            if self.submodel in ["spectral","spatial"]:
                y_pred.append(pred[0])
                y_true.append(label[0])
            else:
                y_pred.append(pred)
                y_true.append(label)       

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        if self.submodel is "metadata":
            name = "Metadata Confusion Matrix"        
        if self.submodel in ["spectral","spatial"]:
            name = "Submodel Confusion Matrix"
        elif self.submodel in ["ensemble"]:
            name = "Ensemble Matrix"
        else:
            name = "Confusion Matrix"

        cm = self.experiment.log_confusion_matrix(
            y_true,
            y_pred,
            title=name,
            file_name= name,
            labels=self.label_names,
            max_categories=77)

class ImageCallback(Callback):

    def __init__(self, experiment, dataset, label_names, submodel=False):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names
        self.submodel = submodel

    def on_train_end(self, epoch, logs={}):
        """Plot sample images with labels annotated"""

        images = []
        y_pred = []
        y_true = []

        #fill until there is atleast 20 images
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
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.00001,
                                  verbose=1)
    callback_list.append(reduce_lr)

    confusion_matrix = ConfusionMatrixCallback(experiment, validation_data, label_names, submodel=submodel)
    callback_list.append(confusion_matrix)
    
    f1 = F1Callback(experiment=experiment, train_dataset=train_data, eval_dataset=validation_data, label_names=label_names, submodel=submodel)
    callback_list.append(f1)
    
    if submodel is None:
        plot_images = ImageCallback(experiment, validation_data, label_names, submodel=submodel)
        callback_list.append(plot_images)
        
    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=10)

    return callback_list
