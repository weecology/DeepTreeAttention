#Callbacks
"""Create training callbacks"""

import os
import numpy as np

from datetime import datetime
from DeepTreeAttention.utils import metrics
from DeepTreeAttention.visualization import visualize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import Callback, TensorBoard


class ConfusionMatrixCallback(Callback):

    def __init__(self, experiment, dataset, label_names):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names

    def on_epoch_end(self, epoch, logs={}):
        y_true = []
        y_pred = []

        for image, label in self.dataset:
            pred = self.model.predict(image)
            y_pred.append(pred)
            y_true.append(label[0])

        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        macro, micro = metrics.f1_scores(y_true, y_pred)
        self.experiment.log_metric("MicroF1", micro)
        self.experiment.log_metric("MacroF1", macro)

        self.experiment.log_confusion_matrix(
            y_true,
            y_pred,
            title="Confusion Matrix, Epoch #%d" % (epoch + 1),
            file_name="confusion-matrix-%03d.json" % (epoch + 1),
            labels=self.label_names)


class ImageCallback(Callback):

    def __init__(self, experiment, dataset, label_names):
        self.experiment = experiment
        self.dataset = dataset
        self.label_names = label_names

    def on_train_end(self, epoch, logs={}):
        """Plot sample images with labels annotated"""

        images = []
        y_pred = []
        y_true = []

        #fill until there is atleast 20 images
        limit = 20
        num_images = 0
        for image, label in self.dataset:
            if num_images < limit:
                pred = self.model.predict(image)
                images.append(image)
                y_pred.append(pred)
                y_true.append(label[0])
                num_images += image.shape[0]
            else:
                break

        images = np.column_stack(images)
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


def create(experiment, validation_data, log_dir=None, label_names=None):
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=5,
                                  min_lr=0.00001,
                                  verbose=1)

    confusion_matrix = ConfusionMatrixCallback(experiment, validation_data, label_names)
    plot_images = ImageCallback(experiment, validation_data, label_names)

    if log_dir is not None:
        tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, profile_batch=10)

    return [reduce_lr, confusion_matrix, plot_images]
