import os
import re
import sys
import pdb
import cv2
import glob
import copy
import time
import math
import random
import shutil
import itertools
import numpy as np
import pandas as pd
from decorators import timer
from natsort import natsorted

@timer
def organize_sample_paths(image_paths, mask_paths, labels=None, savefile=None):
    """
    This function, given a list of image and mask paths, creates a pandas dataframe
    that contains a column for the image paths, and one column per
    label. Each row corresponds to a sample, and holds the paths for this sample's
    image and masks.
    Before loading the data a check is performed, and only the data for which
    both an image and at least one label is available are included in the dataframe.
    This is checked based on the ids that are implied in the file naming (image_<id>).
    If more than one data samples with the same ids exist, this sample is not included
    """
    # infer labels
    if labels is None:
        labels = list(set([os.path.normpath(x).split(os.sep)[-2] for x in mask_paths]))
        labels = natsorted(labels)
    # infer sample id from the image name
    ids = [os.path.splitext(os.path.basename(x))[0] for x in image_paths]
    ids = [id.replace('image_', '') for id in ids]
    paths = {key: list() for key in ['image']+labels}
    for i, id in enumerate(ids):
        img_paths = [p for p in image_paths if re.search(f'\{os.sep}(image_)?{id}\.\w{{2,5}}$', p)]
        if len(img_paths)!=1:
            # ignore sample if more than one samples with the same id exist
            continue
        else:
            paths['image'].append(img_paths[0])
        for label in labels:
            # find mask paths that correspond to the current id
            pattern = f'\{os.sep}{label}\{os.sep}(mask_)?{id}\.\w{{2,5}}$'
            corr_masks = [p for p in mask_paths if re.search(pattern, p)]
            if len(set(corr_masks))==1:
                paths[label].append(corr_masks[0])
            else:
                paths[label].append(None)
    df = pd.DataFrame.from_dict(paths)
    # delete rows with no labels if they exist
    df.dropna(axis=0, how='all', subset=labels, inplace=True)

    if savefile is not None:
        df.to_csv(savefile)

    return df


def data_loader(image_paths, mask_paths, labels=None, width=640, height=480):
    """
    This function, given a list of image and mask paths, returns the images and
    masks in the form of numpy arrays.
    INPUTS:
    @image_paths: a list of image paths
    @mask_paths: a dictionary, with items {label: [list of mask_paths]} for all the
                labels
    @labels: an ordered list of label descriptions. Has to be a subset of the mask
            keys()
    @width: integer, desired width of images in pixels
    @height: integer, desired height of images in pixels
    """
    if labels is None:
        labels = list(mask_paths.keys())
    if 'background' not in labels:
        labels = ['background'] + labels
    #load all images
    images = np.asarray([cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB), (width, height))
                        for p in image_paths])
    # add an extra mask at the beginning for the background class
    masks = np.zeros((images.shape[:-1] + (len(labels),))) # , dtype=int
    for l, label in enumerate(labels[1:]):
        try:
            masks[..., l+1] = np.asarray([cv2.resize(cv2.imread(p, cv2.IMREAD_GRAYSCALE)/255,
                                    (width, height)) for p in mask_paths[label]])
        except:
            # if a mask file does not exist or cannot be loaded, the mask is left
            # black for the moment
            continue
    # only keep data with no black images
    no_black = [i for i in range(len(images)) if np.sum(images[i,...]!=0)]
    images = images[no_black, ...]
    masks = masks[no_black, ...]

    # assign background class to pixels without class
    tmp = np.sum(masks[..., 1:], axis=-1)==0
    masks[..., 0] = tmp.astype(float)
    del tmp

    return images, masks, labels


def sort_and_load_data(image_paths, mask_paths, labels=None, width=640, height=480):
    """
    This function, given a list of image and mask paths, returns the images and
    masks in the form of numpy arrays.
    Before loading the data a check is performed, and only the data for which
    both an image and at least one possible label are available are loaded.
    This is checked based on the ids that are implied in the file naming (image_<id>).
    If more than one data samples with the same ids exist, the sample is ignored.
    """
    df = organize_sample_paths(image_paths, mask_paths, labels=labels)
    image_paths = list(df.image) #ordered image paths
    mask_paths = {key: list(df[key]) for key in df.columns[1:]}# dictionary with ordered mask paths
    imgs, masks, labels = data_loader(image_paths, mask_paths, labels=labels, width=width, height=height)

    return imgs, masks, labels


def clean_dataset(image_paths, mask_paths, coverage=0):
    """
    deletes the images and masks for which the labels take less than img_coverage
    of the entire image
    INPUTS:
    @image_paths: list of image paths
    @mask_paths: list of mask paths
    @coverage: float in [0,1] specifying the ratio of the image that should be
                    covered by positive (not background) classes.
    """
    discard_images = list()
    df = organize_sample_paths(image_paths, mask_paths)
    cnt_corrupted = 0
    for i, row in df.iterrows():
        if cv2.imread(row.image) is None:
            [os.remove(row[key]) for key in row.keys() if row[key] is not None]
            cnt_corrupted += 1
            continue
        img = cv2.cvtColor(cv2.imread(row.image),  cv2.COLOR_BGR2RGB)
        if np.sum(np.isclose(img, np.zeros(img.shape), atol=img.max()*0.01)) >= 0.20*img.size:
            [os.remove(row[key]) for key in row.keys() if row[key] is not None]
            cnt_corrupted +=1
            continue
        masks = np.asarray([cv2.imread(row[key], cv2.IMREAD_GRAYSCALE) for key in row.keys()
                            if key!='image' and row[key] is not None])
        if np.count_nonzero(masks)/masks[0].size < coverage:
            discard_images.append(img)
            [os.remove(row[key]) for key in row.keys() if row[key] is not None]

    print(f'{cnt_corrupted} corrupted images were found (more than 20% black)')
    return discard_images
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
