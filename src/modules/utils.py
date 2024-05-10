import os
import re
import sys
import pdb
import glob
import json
import shutil
import flatdict
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted

from .decorators import forall, verify_format

def create_folders(basefolder: str, *folders):
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


def files_with_extensions(*args, datapath: str, recursive: bool=False):
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
    files = [f for f in files if re.search(pattern, f) and not f.startswith('._')]
    return files


def available_id(savefolder: str, prefix: str='') -> str:
    """
    finds an unused id of the form 'vXXXX' for an experiment. The criterion is
    that no folder '<prefix>_<id>' exists inside savefolder
    """
    i = 0
    while True:
        tmp_id = f'{prefix}v{i:04d}'
        folder = os.path.join(savefolder, f'{tmp_id}')
        i += 1
        if not os.path.exists(folder):
            break
    return tmp_id


def copy_ext(src: str, dest: str) -> None:
    """
    extension of shutil copy2 function to deal with non existing folder.
    If a folder does not exist, this function creates it
    """
    if Path(dest).is_dir():
        os.makedirs(dest, exist_ok=True)
    else:
        os.makedirs(Path(dest).parent, exist_ok=True)

    shutil.copy2(src, dest)


def txt2dict(txt_path: str) -> dict:
    """
    reads a txt file whose each line contains 2 elements, the first being the
    category description (matching the object material name in blender), and the second
    being an integer with the category_id.
    Fills a dictionary with the category descriptions as keys and the ids as values,
    and returns it
    """
    dictio = dict()
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = [x for x in line.split()]
            if line == "\n":
                continue
            dictio[line[0]] = line[1]
    return dictio


def dict2txt(dictio: dict, savepath: str):
    with open(savepath, "w") as f:
        lines = ['{} {}\n'.format(key, value) for key,value in dictio.items()]
        f.writelines(lines)


def dict2csv(dictio: dict, savepath: str):
    """
    saves a dictionary in csv format. The dictionary keys will be the
    columns
    """
    if np.all([type(x) in [list, np.ndarray] for x in dictio.values()]):
        df = pd.DataFrame.from_dict(dictio)
    else:
        df = pd.Series(dictio).to_frame().T
    with open(savepath, mode='a') as f:
        df.to_csv(f, mode='a', float_format="%.3f", header=f.tell()==0)


# def dict2excel(dictio: dict, savepath: str, sheet_name: str='Sheet1'):
#     """
#     saves a dictionary in xlsx format. The dictionary keys will be the
#     columns
#     """
#     if np.all([type(x) in [list, np.ndarray] for x in dictio.values()]):
#         df = pd.DataFrame.from_dict(dictio)
#     else:
#         df = pd.Series(dictio).to_frame().T
#     header = False
#     if not os.path.exists(savepath):
#         with pd.ExcelWriter(savepath) as writer:
#             pd.DataFrame().to_excel(writer)
#         header = True
#     with pd.ExcelWriter(savepath, mode='a', if_sheet_exists='overlay', engine="openpyxl") as writer:
#         pdb.set_trace()
#         # there is a problem here it does not work as expected when appending to
#         # existing file
#         df.to_excel(writer, sheet_name=sheet_name, float_format="%.3f", header=header)
#
#     # if not os.path.exists(excel_file):
#     #     with pd.ExcelWriter(excel_file) as writer:
#     #         pd.DataFrame().to_excel(writer)
#     # book = openpyxl.load_workbook(excel_file)
#     # with pd.ExcelWriter(excel_file, engine='openpyxl', mode='a') as writer:
#     #     writer.book = book
#     #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
#     #     information_radius.to_excel(writer, sheet_name='information_radius_asked_questions_{}mues'.format(len(mues)))


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
