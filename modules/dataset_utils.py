import os
import sys
import pdb
import bpy
import cv2
import glob
import copy
import time
import math
import random
import shutil
import itertools
import numpy as np
from natsort import natsorted


def data_loder(image_paths, mask_paths, labels=None):
    """
    This function, given a list of image and mask paths, returns the images and
    masks in the form of numpy arrays.
    Before loading the data a check is performed, and only the data for which
    both an image and all possible labels are available are loaded.
    This is checked based on the ids that are implied in the file naming (image_<id>).
    If more than one data samples with the same ids exist, this function might
    (for the moment) give arbitrarily wrong results.
    """
    # infer labels
    if labels is None:
        labels = list(set([os.path.normpath(x).split(os.sep)[-2] for x in mask_paths]))
        labels = natsorted(labels)
    if 'background' not in labels:
        labels = ['background'] + labels

    #clean paths so as to contain only complete data
    ids = [os.path.splitext(os.path.basename(x))[0] for x in image_paths]
    ids = [id.replace('image_', '') for id in ids]
    ids_with_label = dict.fromkeys(labels[1:])
    for label in labels[1:]:
        ids_with_label[label] = [i for i in ids
                                for path in mask_paths
                                if ((label in path) and (i in path))]
    # keep only ids that have all their labels
    ids = set(ids).intersection(*ids_with_label.values())
    paths = {key: list() for key in ['image']+labels[1:]}
    for id in ids:
        paths['image'].extend([p for p in image_paths if id in p])
        for label in labels[1:]:
            paths[label].extend([p for p in mask_paths
                                    if (id in p) and (label in p)])

    # load image data
    images = np.asarray([cv2.cvtColor(cv2.imread(p),  cv2.COLOR_BGR2RGB)
                        for p in paths['image']])
    masks = np.zeros(images.shape[:3] + (len(labels),))
    for i, label in enumerate(labels[1:]):
        masks[..., i+1] = np.asarray([cv2.imread(p, cv2.IMREAD_GRAYSCALE)/255
                                    for p in paths[label]]) #, cv2.IMREAD_GRAYSCALE

    # assign background class to pixels without class
    tmp = np.sum(masks[..., 1:], axis=-1)==0
    masks[..., 0] = tmp.astype(float)
    del tmp

    return images, masks, labels


def clean_dataset(image_paths, mask_paths, img_coverage=0, save_path=None):
    """
    deletes the images and masks for which the labels take less than img_coverage
    of the entire image
    INPUTS:
    @image_paths: list of image paths
    @mask_paths: list of mask paths
    @img_coverage: float in [0,1] specifying the ratio of the image that should be
                    covered by positive (not background) classes.
    """
    discard_images = list()
    # if img_coverage==0:
    #     return discard_images
    for im_path in image_paths:
        id = os.path.splitext(os.path.basename(im_path))[0].replace('image_', '')
        relevant_mask_paths = [x for x in mask_paths if id in x]
        # sometimes blender produces all black images, so the following check is made here
        if math.isclose(np.sum(cv2.cvtColor(cv2.imread(im_path),  cv2.COLOR_BGR2RGB)), 0):
            os.remove(im_path)
            [os.remove(x) for x in relevant_mask_paths]
            continue
        masks = np.asarray([cv2.imread(x, cv2.IMREAD_GRAYSCALE) for x in relevant_mask_paths])
        if np.sum(masks>0)/masks[0].size<img_coverage:
            discard_images.append(cv2.cvtColor(cv2.imread(im_path),  cv2.COLOR_BGR2RGB))
            os.remove(im_path)
            [os.remove(x) for x in relevant_mask_paths]
    return discard_images
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
