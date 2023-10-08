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
import shutil
import argparse
import numpy as np
from pathlib import Path

sys.path.append('..')
import modules.utils as utils
import modules.dataset_utils as dts

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


def split_real(data_dir: str) -> None:
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
