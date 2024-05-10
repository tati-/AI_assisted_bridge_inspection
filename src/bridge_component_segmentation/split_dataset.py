"""
This script takes as argument a folder path containing bridge images,
and identifies the number of bridges. It then does a bridge-based
train-validation-test set split (60-20-20) and saves the data and labels in
3 different folders, train, validation, test

NOTE! the script supposes that the image name for the synthetic data contains
the string '_bridge<num>_', and that the data are organized, in the original
folder in an 'images' and a 'masks' folder. For the real dataset, he mask filenames
should be either exactly the same as the image ones, either have a 'mask_' prefix

example usage: python split_dataset.py -dp /path/to/data
"""
import os
import re
import sys
import pdb
import glob
import math
import random
import shutil
import argparse
import numpy as np
from pathlib import Path

sys.path.append('..')
import modules.utils as utils
import modules.dataset_utils as dts

same_bridge_real = [['DSC00019', 'DSC00025'],
                    ['DSC00028', 'DSC00029'],
                    ['DSC00058', 'DSC00060'],
                    ['DSC00104', 'DSC00108'],
                    ['DSC00329', 'DSC00333'],
                    ['DSC00399', 'DSC00400', 'DSC00403'],
                    ['DSCF0058', 'DSCF0059'],
                    ['DSCN3840', 'DSCN3842'],
                    ['DSCN3845', 'DSCN3853', 'DSCN3859'],
                    ['DSCN3873', 'DSCN3876'],
                    ['DSCN3882', 'DSCN3883', 'DSCN3886', 'DSCN3888', 'DSCN3890'],
                    ['DSCN3905', 'DSCN3906', 'DSCN3920', 'DSCN3926'],
                    ['DSCN3910', 'DSCN3916'], # good for test
                    ['DSCN4717', 'DSCN4718', 'DSCN4719', 'DSCN4724', 'DSCN4726', 'DSCN4727'],
                    ['DSCN8854_redimensionner', 'DSCN8855_redimensionner'],
                    ['IMG_0295', 'IMG_0296', 'IMG_0297'],
                    ['IMG_0421', 'IMG_0422'],
                    ['IMG_0424', 'IMG_0451', 'IMG_4174', 'IMG_4175'],
                    ['IMG_0478', 'IMG_0492'],
                    ['IMG_0499', 'IMG_1493'], # good for test
                    ['IMG_2556', 'IMG_2548'],
                    ['IMG_3076', 'IMG_3071'],
                    ['IMG_3135', 'IMG_3136', 'IMG_3141'],
                    ['IMG_3172', 'IMG_3148', 'IMG_3175'],
                    ['IMG_3176', 'IMG_3180', 'IMG_3181-Modif-VVV', 'IMG_3186'],
                    ['IMG_4753', 'IMG_4757', 'IMG_4794', 'IMG_4797'],
                    ['P1012596', 'P1012599'],
                    ['P1020687', 'P1020773'],
                    ['P1070019', 'P1070021'],
                    ['P1080029', 'P1080035'],
                    ['P1090160', 'P1090161']
                    ]

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-dp', help='data path', type=str, required=True)
    arg('--synthetic', help='If set denotes that the dataset used is synthetic', action='store_true')

    args = parser.parse_args()
    return args


def split_synthetic(data_dir: str) -> None:
    # find image paths (a folder named images is expected to exist inside data_dir)
    image_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                        datapath=os.path.join(data_dir, 'images'))
    if not image_paths:
        pass
    mask_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                        datapath=os.path.join(data_dir, 'masks'),
                                        recursive=True)
    # infer unique bridge identifiers
    bridges_ids = dts.unique_bridges(image_paths)
    # split to train and test
    bridges = dict()
    bridges['train'] = bridges_ids[:int(0.6*len(bridges_ids))]
    bridges['validation'] = bridges_ids[int(0.6*len(bridges_ids)):int(0.8*len(bridges_ids))]
    bridges['test'] = bridges_ids[int(0.8*len(bridges_ids)):]
    # seperate in folders by set
    for key in bridges.keys():
        img_paths = [x for x in image_paths if
                        any([f'{br}_' in Path(x).stem for br in bridges[key]])]
        msk_paths = [x for x in mask_paths if
                        any([f'{br}_' in Path(x).stem for br in bridges[key]])]
        for p in img_paths:
            new_p = p.replace(f'{os.sep}images{os.sep}', f'{os.sep}{key}{os.sep}images{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)
        for p in msk_paths:
            new_p = p.replace(f'{os.sep}masks{os.sep}', f'{os.sep}{key}{os.sep}masks{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)

    # delete the original image and mask folders if empty
    if  not any(Path(data_dir).joinpath('images').iterdir()):
        shutil.rmtree(Path(data_dir).joinpath('images'))
    if  not any(Path(data_dir).joinpath('masks').iterdir()):
        shutil.rmtree(Path(data_dir).joinpath('masks'))


def split_real_naive(data_dir: str) -> None:
    """
    splits a real dataset to training, validation, and testing set, assuming all
    images come from different bridges
    """
    # find image paths (a folder named images is expected to exist inside data_dir)
    image_paths  = [str(p) for p in Path(os.path.join(data_dir, 'images')).rglob('*.*')
                    if (p.suffix in ['.jpg', '.JPG', '.png']
                    and not os.path.basename(p).startswith('._'))]
    if not image_paths:
        pass
    mask_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                        datapath=os.path.join(data_dir, 'masks'),
                                        recursive=True)
    img_ids = [Path(x).stem for x in image_paths]
    # split to train and test
    ids = dict()
    ids['train'] = img_ids[:int(0.6*len(img_ids))]
    ids['validation'] = img_ids[int(0.6*len(img_ids)):int(0.8*len(img_ids))]
    ids['test'] = img_ids[int(0.8*len(img_ids)):]

    # seperate in folders by set
    for key in ids.keys():
        img_paths = [x for x in image_paths if Path(x).stem in ids[key]]
        msk_paths = [x for x in mask_paths if any([re.search(f'^(mask_)?{id}$', Path(x).stem) for id in ids[key]])]
        for p in img_paths:
            new_p = p.replace(f'{os.sep}images{os.sep}', f'{os.sep}{key}{os.sep}images{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)
        for p in msk_paths:
            new_p = p.replace(f'{os.sep}masks{os.sep}', f'{os.sep}{key}{os.sep}masks{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)

    # delete the original image and mask folders if empty
    if  not any(Path(data_dir).joinpath('images').iterdir()):
        shutil.rmtree(Path(data_dir).joinpath('images'))
    if  not Path(data_dir).joinpath('masks').rglob('*.*'):
        shutil.rmtree(Path(data_dir).joinpath('masks'))


def split_real(data_dir: str) -> None:
    """
    splits a real dataset by taking into account the fact that some images
    are from the same bridge, makes sure to put all those in the same set
    """
    # find image paths (a folder named images is expected to exist inside data_dir)
    image_paths  = [str(p) for p in Path(os.path.join(data_dir, 'images')).rglob('*.*')
                    if (p.suffix in ['.jpg', '.JPG', '.png']
                    and not os.path.basename(p).startswith('._'))]
    if not image_paths:
        pass
    mask_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                        datapath=os.path.join(data_dir, 'masks'),
                                        recursive=True)
    img_ids = [Path(x).stem for x in image_paths]
    # split to train and test
    ids = {key: list() for key in ['train', 'validation', 'test']}
    n_samples = dict()
    n_samples['train'] = int(0.6*len(img_ids))
    n_samples['validation'] = int(0.2*len(img_ids))
    n_samples['test'] = math.ceil(0.2*len(img_ids))
    for key in ids.keys():
        while len(ids[key])<n_samples[key] and len(img_ids)>0:
            to_add = [random.choice(img_ids)]
            # check if selected image belongs to a set of images of the same bridge
            for bridge_set in same_bridge_real:
                if to_add[0] in bridge_set:
                    to_add = bridge_set
                    break
            ids[key].extend(to_add)
            # remove already assigned elements from list of candidates
            img_ids = [id for id in img_ids if id not in to_add]

    # seperate in folders by set
    for key in ids.keys():
        img_paths = [x for x in image_paths if Path(x).stem in ids[key]]
        msk_paths = [x for x in mask_paths if any([re.search(f'^(mask_)?{id}$', Path(x).stem) for id in ids[key]])]
        for p in img_paths:
            new_p = p.replace(f'{os.sep}images{os.sep}', f'{os.sep}{key}{os.sep}images{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)
        for p in msk_paths:
            new_p = p.replace(f'{os.sep}masks{os.sep}', f'{os.sep}{key}{os.sep}masks{os.sep}')
            shutil.move(p, new_p, copy_function=utils.copy_ext)

    # delete the original image and mask folders if empty
    if  not any(Path(data_dir).joinpath('images').iterdir()):
        shutil.rmtree(Path(data_dir).joinpath('images'))
    if  not Path(data_dir).joinpath('masks').rglob('*.*'):
        shutil.rmtree(Path(data_dir).joinpath('masks'))


def split(data_dir: str, synthetic: bool=False) -> None:
    if synthetic:
        return split_synthetic(data_dir)
    else:
        return split_real(data_dir)


if __name__ == "__main__":
    args = get_args()
    split(args.dp, args.synthetic)
"""
############################################################################
                                END
############################################################################
"""
