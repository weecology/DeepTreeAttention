import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score


def f1_scores(y_true, y_pred):
    """Calculate micro, macro
    Args:
        y_true: one_hot ground truth labels
        y_pred: softmax classes 
    Returns:
        macro: macro average fscore
        micro: micro averge fscore
    """
    #F1 scores
    y_true_integer = np.argmax(y_true, axis=1)
    y_pred_integer = np.argmax(y_pred, axis=1)

    macro = f1_score(y_true_integer, y_pred_integer, average='macro')
    micro = f1_score(y_true_integer, y_pred_integer, average='micro')

    return macro, micro


def confusion(y_true, y_pred, num_classes):
    confusion = tf.math.confusion_matrix(labels=y_true,
                                         predictions=y_pred,
                                         num_classes=num_classes)
    return confusion
