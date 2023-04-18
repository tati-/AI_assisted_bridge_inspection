"""
This scripts generates a number of bridge 3d models along with their annotation
starting from a .json file with all necessary parameters
The annotation is saved in the form of different materials for different classes
in an obj file.
# sample usage remote: python generate_3d_models.py -basefile ../../data/Bridge_parametric_models/PIPO/base-1traverse.json -params ../../data/Bridge_parametric_models/PIPO/parameter_set-1traverse.json  -bridges 2
# sample usage local: python generate_3d_models.py -basefile ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/base-1traverse.json -params ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/parameter_set-1traverse.json -bridges 10

NOTE: for the moment this code is similar to generate_3d_semantic_models.py, but it
        does not do the extra step of creating the blender file with the seperate
        object per category. This is a temporary solution to have a version that
        does not need the bpy package installed. In the future the two functionalities
        should be decoupled (json->obj, obj->blender), and therefore the
        generate_3d_semantic_models.py script functionality will be split in this script
        and another one to be made
"""
import os
import sys
import pdb
import glob
import copy
import math
import time
import random
import argparse
import warnings
import datetime
import numpy as np
# import pandas as pd
sys.path.append('modules')
sys.path.append('classes')
import utils
from bridge import BridgeModel as Bridge


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-basefile', help='input file path [acceptable formats: .json]', type=str, required=True)
    arg('-params', help='json file path with parameters', type=str)
    arg('-savefolder', help= 'folder to save generated dataset [default: same as input file directory]', type=str)
    arg('-bridges', help='number of bridges [default: 1]', type=int, default=1)
    args = parser.parse_args()
    return args


def generate_bridge_models(basefile, params=None, savefolder=None, bridges=1):
    savefolder = os.path.dirname(basefile) if savefolder is None else savefolder
    assert os.path.splitext(basefile)[1]=='.json', 'No input file of acceptable type was given [.json]'
    paths = utils.create_folders(savefolder, 'obj')

    # base numbering on the blender file
    start_ind = utils.last_file_index(
            glob.glob(os.path.join(paths['obj'], 'bridge*.obj'))) + 1

    for bridge in range(bridges):
        bridge_id = bridge+start_ind

        ############################################################################
        #                LOAD AND INITIALISE MESH                                  #
        ############################################################################
        obj_path = os.path.join(paths['obj'], f'bridge{bridge_id}.obj')
        mtl_path = os.path.join(paths['obj'], f'bridge{bridge_id}.mtl')
        bridge_model = Bridge(json_data_path=basefile,
                obj_path=obj_path,
                label_path=mtl_path,
                json_param_path=params)
        bridge_model.create_mesh()
        bridge_model.save_mesh()
        del bridge_model


if __name__ == "__main__":
    start = time.time()
    args = get_args()
    generate_bridge_models(**vars(args))
    print('Total time elapsed: {}'.format(datetime.timedelta(seconds=time.time()-start)))

    """
    ############################################################################
                                    END
    ############################################################################
    """
