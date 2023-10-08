"""
This script creates a file with the number of parameters for different models,
for documentation purposes
example usage: python model_complexity.py -csv ../../models/complexity_stats.csv

"""
import os
import sys
import pdb
import time
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('..')
exec(open('../modules/ml_imports.py').read())
import modules.ml_utils as ml
from modules.decorators import timer

architectures = ['Unet_reduced', 'Unet', 'Linknet', 'FPN', 'PSPNet']


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-csv', type=str, required=True, help='csv file path to save the stats to')
    arg('-insize', help='input size, used to test inference time', type=int, nargs=2, default=[640, 480])

    args = parser.parse_args()
    return args


@timer
def model_comparison(csv, insize=(640, 480)):
    """
    this function creates a csv file with the number of parameters for a number
    of model architectures and backbone combinations.
    INPUTS:
    @csv: path to csv file
    @insize: tuple, (width, height)
    """
    time_col_name = f'infer_time_{insize[0]}x{insize[1]}_s'
    columns = ['model', 'backbone', 'trainable', 'non_trainable', 'total', time_col_name]
    model_parameters = {key: list() for key in columns}

    input_shape = (insize[1], insize[0], 3)
    # random array with the given shape, to test inference time
    img = np.random.rand(*input_shape)
    # if input shape is not divisible by 48 the pspnet needs special treatment
    if input_shape[0]%48!=0 or input_shape[1]%48!=0:
        pspnet_flag = True

    for arch in architectures[:2]:
        # define classifier
        model = ml.get_network(arch, input_shape=input_shape)
        # get info about model size
        tr, non_tr = ml.model_stats(model)
        # update dictionary
        model_parameters['model'].append(arch)
        model_parameters['backbone'].append(None)
        model_parameters['trainable'].append(tr)
        model_parameters['non_trainable'].append(non_tr)
        model_parameters['total'].append(tr+non_tr)
        # test inference time
        start_pred = time.process_time()
        _ = model.predict(img[np.newaxis, ...])
        pred_time = time.process_time() - start_pred
        model_parameters[time_col_name].append(f'{pred_time:.2f}')
    # semantic segmentation package models
    for arch in architectures[1:]:
        for backbone in ml.sm_backbones:
            # define classifier
            if arch=='PSPNet' and pspnet_flag:
                model = ml.get_network(arch, backbone)
            else:
                model = ml.get_network(arch, backbone, input_shape=input_shape)
            # get info about model size
            tr, non_tr = ml.model_stats(model)
            # update dictionary
            model_parameters['model'].append(arch)
            model_parameters['backbone'].append(backbone)
            model_parameters['trainable'].append(tr)
            model_parameters['non_trainable'].append(non_tr)
            model_parameters['total'].append(tr+non_tr)
            # test inference time
            if arch=='PSPNet' and pspnet_flag:
                model_parameters[time_col_name].append(None)
            else:
                start_pred = time.process_time()
                _ = model.predict(img[np.newaxis, ...])
                pred_time = time.process_time() - start_pred
                model_parameters[time_col_name].append(f'{pred_time:.2f}')

    #write to csv file
    params = pd.DataFrame.from_dict(model_parameters)
    params.to_csv(csv)
    print(f'Info saved in {csv}')


if __name__ == "__main__":
    args = get_args()
    if Path(args.csv).suffix!='.csv':
        sys.exit('CSV file name needed as input!')
    model_comparison(**vars(args))
"""
############################################################################
                                END
############################################################################
"""
