"""
This scripts refines a set of images (cleans the dataset so that only images
that contain a significant amount of labels stay)
# sample usage: python refine_dataset.py -d path/to/image_directory -cov 0.1
"""
import os
import sys
import pdb
import cv2
import glob
import copy
import math
import time
import random
import shutil
import argparse
import warnings
import datetime
import unidecode
import numpy as np

sys.path.append('..')
import modules.utils as utils
import modules.visualization as vis
import modules.dataset_utils as dts
from modules.decorators import timer


def restricted_float(x):
    """
    defines a float that is limited to be between 0 and 1
    """
    if x is None:
        return x
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} not a floating-point literal')
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError(f'{x} not in range [0, 1]')
    return x


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-d', '-datapath', help='folder containing the dataset', type=str,  required=True)
    arg('-cov', help='minimum coverage, percentage of image that contains the structure, for the image to be kept in the dataset [default:0.1]', type=restricted_float, default=0.1)
    args = parser.parse_args()
    return args


@timer
def discard_unrelevant(datapath, coverage=0.1):
    """
    discard all images inside 'images' folder of datapath, that contain less
    that coverage positive classes.
    Also creates an image as a small demo of the discarded images
    """
    images_discarded = list()
    image_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(datapath, 'images'))
    mask_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(datapath, 'masks'), recursive=True)
    discarded = dts.clean_dataset(image_paths, mask_paths, coverage)
    if len(discarded)>0:
        images_discarded=random.sample(discarded, min(5, len(discarded)))
    # create a plot with some example discarded images
    if len(images_discarded)>0:
        path = utils.create_folders(datapath, 'demo')[0]
        vis.demo_discarded(images_discarded, coverage, path)

    # in case there is no image left after the cleaning, delete folder
    if len(images_discarded)==len(image_paths):
        print("Not a single acceptable image was produced, so dataset {} " \
        "is removed.\n".format(os.path.normpath(datapath).split(os.sep)[-1]))
        shutil.rmtree(datapath)
        sys.exit()
    else:
        info_path = os.path.join(datapath, 'info.txt')
        with open(info_path, 'a') as f:
            f.write(f'Usable images: {len(image_paths)-len(images_discarded)}\n\n')


if __name__ == "__main__":
    args = get_args()
    discard_unrelevant(args.d, args.cov)
    """
    ############################################################################
                                    END
    ############################################################################
    """
