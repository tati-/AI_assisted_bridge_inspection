import os
import re
import sys
import cv2
import glob
import copy
import warnings
import numpy as np
import pandas as pd
from natsort import natsorted

from .decorators import timer, forall


@forall
def adjust_img_range(image, img_range: tuple=(0,1)) -> list:
    """
    this function makes sure that the pixel values of the images
    lie in the appropriate range
    NOTE: for the moment only works for the (0,1) and (0, 255) cases,
            not for arbitrary ranges
    """
    if img_range==(0,1):
        return image/(np.max(image)+1e-5)
    elif img_range==(0,255):
        return (image/(np.max(image)+1e-5)*255).astype(int)
    else:
        warnings.warn(f'\n##-----## \nWarning: Image range {img_range} '\
                        'not recognised, image is returned as such.\n##-----##')
        return image


def unique_bridges(image_paths: list) -> list:
    """
    given a list of image paths returns a list with the unique
    bridge identifiers. (which implies the number of seperate bridges)
    NOTE: this function supposes that the image files are named
            <whatever>_bridge<#>_xxxx.png.
    """
    filenames = [os.path.basename(x) for x in image_paths]
    tmp = [x[x.find('bridge'):] for x in filenames]
    tmp = [x[:x.find('_')] for x in tmp]
    bridges_ids = list(set([x for x in tmp]))

    return bridges_ids


def clean_dataset(image_paths: list,
                mask_paths: list,
                coverage=0):
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
