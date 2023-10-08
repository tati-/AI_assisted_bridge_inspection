import os
import re
import sys
import pdb
import math
import random
import inspect
import warnings
import numpy as np
from scipy import stats
# import numpy.typing as npt
import matplotlib.pyplot as plt
import segmentation_models as sm
from sklearn.metrics import confusion_matrix, f1_score

from . import utils
from .decorators import optional, timer
# exec(open('modules/ml_imports.py').read())


def tp_from_cm(confusion_matrix) -> int:
    """
    get the mumber of true positives from a square confusion matrix
    """
    return np.sum(np.diagonal(confusion_matrix))


def IoU_from_cm(confusion_matrix, mean=False):
    """
    get IoU from the confusion matrix
    INPUTS:
    @confusion_matrix: square numpy array where rows correspond to real labels
                       and columns to predicted labels
    @mean: boolean, if true the mean over all classes is returned. If false, a list
            with the per class values is returned
    """
    assert confusion_matrix.ndim==2 and confusion_matrix.shape[0] == confusion_matrix.shape[1], f'A square 2d matrix is expected as input to meanIoU, instead {confusion_matrix.shape} was given'
    iou = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[i,i]
        false_positives = np.sum(np.delete(confusion_matrix[:, i], i))
        false_negatives = np.sum(np.delete(confusion_matrix[i, :], i))
        iou[i] = true_positives / (true_positives + false_negatives + false_positives)
    if mean:
        return np.mean(iou)
    else:
        return iou


def precision_from_cm(confusion_matrix, mean=False):
    """
    get precision from the confusion matrix
    INPUTS:
    @confusion_matrix: square numpy array where rows correspond to real labels
                       and columns to predicted labels
    @mean: boolean, if true the mean over all classes is returned. If false, a list
            with the per class values is returned
    """
    assert confusion_matrix.ndim==2 and confusion_matrix.shape[0] == confusion_matrix.shape[1], f'A square 2d matrix is expected as input to meanIoU, instead {confusion_matrix.shape} was given'
    precision = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[i,i]
        false_positives = np.sum(np.delete(confusion_matrix[:, i], i))
        precision[i] = true_positives / (true_positives + false_positives)
    precision = np.nan_to_num(precision)
    if mean:
        return np.mean(precision)
    else:
        return precision


def recall_from_cm(confusion_matrix, mean=False):
    """
    get recall from the confusion matrix
    INPUTS:
    @confusion_matrix: square numpy array where rows correspond to real labels
                       and columns to predicted labels
    @mean: boolean, if true the mean over all classes is returned. If false, a list
            with the per class values is returned
    """
    assert confusion_matrix.ndim==2 and confusion_matrix.shape[0] == confusion_matrix.shape[1], f'A square 2d matrix is expected as input to meanIoU, instead {confusion_matrix.shape} was given'
    recall = np.zeros(confusion_matrix.shape[0])
    for i in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[i,i]
        false_negatives = np.sum(np.delete(confusion_matrix[i, :], i))
        recall[i] = true_positives / (true_positives + false_negatives)
    recall = np.nan_to_num(recall)
    if mean:
        return np.mean(recall)
    else:
        return recall


def fscore_from_cm(confusion_matrix, mean=False):
    """
    f1 score from confusion matrix
    INPUTS:
    @confusion_matrix: square numpy array where rows correspond to real labels
                       and columns to predicted labels
    @mean: boolean, if true the mean over all classes is returned. If false, a list
            with the per class values is returned
    """
    assert confusion_matrix.ndim==2 and confusion_matrix.shape[0] == confusion_matrix.shape[1], f'A square 2d matrix is expected as input to meanIoU, instead {confusion_matrix.shape} was given'
    f1 = np.zeros(confusion_matrix.shape[0])
    precision = precision_from_cm(confusion_matrix)
    recall = recall_from_cm(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        try:
            f1[i] = stats.hmean([precision[i], recall[i]])
        except:
            continue
    if mean:
        return np.mean(f1)
    else:
        return f1
