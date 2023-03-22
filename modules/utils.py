import os
import re
import sys
import pdb
import glob
import copy
import math
import time
import random
import shutil
import datetime
import unidecode
import itertools
import numpy as np
from natsort import natsorted
from scipy import interpolate


def create_folders(basefolder, *folders):
    """
    creates a number of subfolders inside a basefolder
    """
    paths = list()
    for folder in folders:
        paths.append(os.path.join(basefolder, folder))
        # if os.path.isdir(path): shutil.rmtree(path)
        os.makedirs(paths[-1], exist_ok=True)

    return paths


def last_file_index(filenames):
    """
    this function takes a list of files assumingly ending with a number.
    That number is considered to be an incremental index. The highest such
    index is then returned as an integer. The part before the index is
    considered to be identical in the files, if not the behavior of the
    sorting algorithm can be unpredictable.
    Example: for files lalal1.txt, lalal2.txt, lalal3.txt, 3 will be returned.
    """
    if len(filenames)==0: return -1;
    filename = natsorted(filenames)[-1]
    match = re.search('(\d+).[a-zA-Z]{1,9}$', filename)
    if match.groups()[0] is None:
        return -1
    else:
        return int(match.groups()[0])


def files_with_extensions(*args, datapath, recursive=False):
    """
    returns a list of files inside the datapath that have one of the
    extensions
    """
    pattern = '\w+\.('
    for arg in args:
        pattern = f'{pattern}{arg}|'
    pattern = pattern[:-1]
    pattern = f'{pattern})$'
    files = glob.glob(os.path.join(datapath, '*.*'), recursive=recursive)
    files = [f for f in files if re.search(pattern, f)]
    return files


def dataset_id(savefolder, prefix='dataset'):
    """
    finds an unused id of the form 'vXXXX' for an experiment. The criterion is
    that no folder '<prefix>_<id>' exists inside savefolder
    """
    i = 0
    while True:
        dts_id = f'v{i:04d}'
        folder = os.path.join(savefolder, f'{prefix}_{dts_id}')
        i += 1
        if not os.path.exists(folder):
            break
    return dts_id


# def synthetic_dataset_paths(savefolder):
#     """
#     creates a folder for a dataset inside savefolder and, some
#     relevant subfolders in it
#     """
#     paths = dict()
#     dts_id = dataset_id(savefolder)
#     paths['base'] = os.path.join(savefolder, f'dataset_{dts_id}')
#     paths['demo'] = os.path.join(paths['base'], 'demo')
#     paths['overview'] = os.path.join(paths['base'], 'overview')
#     paths['info'] = os.path.join(paths['base'], 'info.txt')
#     for path in paths.values():
#         if os.path.isdir(path): shutil.rmtree(path)
#         if os.path.splitext(path)[1]=='':
#             os.makedirs(path, exist_ok=True)
#
#     return paths


def txt2dict(txt_path):
    """
    reads a txt file whose each line contains 2 elements, the first being the
    category description (matching the object material name in blender), and the second
    being an integer with the category_id.
    Fills a dictionary with the category descriptions as keys and the ids as values,
    and returns it
    """
    category_dict = dict()
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = [x for x in line.split()]
            if line == "\n":
                continue
            category_dict[line[0]] = int(line[1])
    return category_dict
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
