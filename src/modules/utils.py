import os
import re
import sys
import pdb
import glob
import json
import shutil
import flatdict
import numpy as np
from natsort import natsorted

from .decorators import forall, verify_format

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
    files = glob.glob(os.path.join(datapath, '**'), recursive=recursive)
    # if recursive:
    #     files = [str(x) for x in Path(datapath).rglob('*')]
    # else:
    #      files = [str(x) for x in Path(datapath).glob('*')]
    files = [f for f in files if re.search(pattern, f)]
    return files


def available_id(savefolder, prefix='dataset'):
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


def all_dict_values(dictio: dict):
    """
    returns all values of a possibly nested dictionary
    """
    d = flatdict.FlatDict(dictio, delimiter='.')
    return d.values()


def lower_ext(x):
    """
    extension of lower() function to leave the input unchanged if it is not a string
    """
    try:
        return x.lower()
    except:
        return x


@forall
def dict_lowercase(dictio: dict, keys_only: bool=False) -> dict:
    """
    lowercase all elements of a list of potentially nested dictionaries.
    if keys_only is set, only lowercase the keys
    """
    d = flatdict.FlatDict(dictio)
    if keys_only:
        d = {lower_ext(k): v for k, v in d.items()}
    else:
        d = {lower_ext(k): lower_ext(v) for k, v in d.items()}
    # reconvert d to FlatDict so that the to_dict function can re transform it to
    # a nested dictionary
    return flatdict.FlatDict(d).as_dict()


@verify_format('.json')
def json_lowercase(json_file: str):
    """
    loads a json file and lowercases all its values and keys.
    """
    with open(json_file) as f:
        string = json.dumps(json.load(f))
    string = string.lower()
    return json.loads(string)


def numpy2native(var):
    """
    this function takes a variable, and if it is a numpy data format
    converts it to native python format (useful to then dump data into a
    json file, since json does not accept numpy datatypes)
    """
    if 'numpy' in str(type(var)):
        try:
            if 'int' in str(type(var)):
                return int(var)
            elif 'float' in str(type(var)):
                return float(var)
            elif 'ndarray' in str(type(var)):
                return var.tolist()
        except:
            return var
    else:
        return var
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
