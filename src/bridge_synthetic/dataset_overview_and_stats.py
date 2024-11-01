"""
This scripts refines a set of images (cleans the dataset so that only images
that contain a significant amount of labels stay)
# sample usage: python dataset_overview_and_stats.py -d path/to/data/dir
"""
import os
import re
import sys
import cv2
import glob
import copy
import time
import random
import shutil
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd

sys.path.append('..')
import modules.utils as utils
import modules.visualization as vis
import modules.dataset_utils as dts
from modules.decorators import optional, timer


@optional
def mist_demo(img_paths, savefolder='.', clear=True):
    """
    this function gets a number of images, and if the corresponding images
    without the mist effect exists, it creates a collective figure showing
    some images without mist on the top, and some images with mist at the
    bottom
    """
    images_mist, images_no_mist = list(), list()
    flag_no_mist = False
    # randomly sample some images and save mist and no mist version
    img_mist_paths = random.sample(img_paths, min(5, len(img_paths)))
    for path in img_mist_paths:
        # assuming there are images without mist saved in a folder
        # 'images_without_mist' next to the images folder
        img_no_mist_path = path.replace(f'{os.sep}images{os.sep}',
                                        f'{os.sep}images_without_mist{os.sep}')
        if os.path.exists(img_no_mist_path):
            flag_no_mist = True
            images_mist.append(cv2.cvtColor(cv2.imread(path),  cv2.COLOR_BGR2RGB))
            images_no_mist.append(cv2.cvtColor(cv2.imread(img_no_mist_path),  cv2.COLOR_BGR2RGB))
    if clear and flag_no_mist:
        shutil.rmtree(re.sub('images$', 'images_without_mist', os.path.dirname(img_paths[0])))
    ############################################################################
    #                               DEMOS                                      #
    ############################################################################
    if len(images_mist)>0:
        vis.demo_mist(images_mist, images_no_mist, n_images=min(len(images_mist), 5),
                        savefolder=savefolder)


@timer
def dataset_overview(image_paths, mask_paths, savefolder='.'):
    """
    this function gets a number of image and mask paths and saves figures depicting
    the images side by side with their segmentation groundtruth
    """
    df = dts.organize_sample_paths(image_paths, mask_paths)
    labels = ['background'] + list(df.keys()[1:])

    class_concentration = vis.inspect_dataset(*[row for i,row in df.iterrows()], labels=labels, savefolder=savefolder)
    class_concentration = pd.DataFrame.from_records(class_concentration)
    class_totals = class_concentration.sum(axis='index')
    class_totals.name = 'n_pixels'
    class_freq = class_totals.divide(class_totals.sum())
    class_freq.name = 'freq'
    class_stats = pd.concat([class_totals, class_freq], axis=1)

    # save number of pixels per class in csv
    csv_path = os.path.join(savefolder, 'class_concentration.csv')
    breakpoint()
    print(class_stats)
    class_stats.to_csv(csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arg = parser.add_argument('-d', '-datapath', help='folder containing the dataset', type=str,  required=True)
    args = parser.parse_args()
    image_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(args.d, '**', 'images'), recursive=True)
    mask_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(args.d, '**', 'masks'), recursive=True)
    # in case there is no image, delete folder
    if len(image_paths)==0:
        print(f'Not a single acceptable image was produced, dataset {args.d} is removed.\n')
        shutil.rmtree(args.d)
        sys.exit()
    demo_folder, overview_folder = utils.create_folders(args.d, 'demo', 'overview')
    ############################################################################
    #                             DEMO MIST VS NO MIST                         #
    ############################################################################
    mist_demo(image_paths, demo_folder, clear=True)

    ############################################################################
    #                         LOAD AND SHOW IMAGES                             #
    ############################################################################
    dataset_overview(image_paths, mask_paths, overview_folder)
    """
    ############################################################################
                                    END
    ############################################################################
    """
