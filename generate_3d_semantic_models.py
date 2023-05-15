"""
This scripts generates a number of bridge 3d models along with their annotation
starting from a .json file with all necessary parameters
# sample usage remote: python generate_3d_semantic_models.py -basefile ../../data/Bridge_parametric_models/PIPO/base-1traverse.json -params ../../data/Bridge_parametric_models/PIPO/parameter_set-1traverse.json -cl ../../data/semantic_classes -bridges 2
# sample usage local: python generate_3d_semantic_models.py -basefile ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/base-1traverse.json -params ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/parameter_set-1traverse.json -cl ~/mounts/deepmachine/MIRAUAR/data/semantic_classes -bridges 10
"""
import os
import sys
import pdb
import bpy
import glob
import copy
import math
import time
import bmesh
import random
import shutil
import argparse
import warnings
import datetime
import unidecode
import numpy as np
# import pandas as pd
sys.path.append('modules')
sys.path.append('classes')
import utils
import blender_utils as bl
from decorators import timer
from bridge import BridgeModel as Bridge


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-basefile', help='input file path [acceptable formats: .json]', type=str, required=True)
    arg('-params', help='json file path with parameters', type=str)
    arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)
    arg('-savefolder', help= 'folder to save generated dataset [default: same as input file directory]', type=str)
    arg('-bridges', help='number of bridges [default: 1]', type=int, default=1)
    args = parser.parse_args()
    return args


@timer
def generate_bridge_models(basefile, params=None, savefolder=None, cl=None, bridges=1):
    savefolder = os.path.dirname(basefile) if savefolder is None else savefolder
    assert os.path.splitext(basefile)[1]=='.json', 'No input file of acceptable type was given [.json]'
    obj_path, blender_path = utils.create_folders(savefolder, 'obj', 'blender')

    # labels_info = pd.DataFrame(data={'id': cat_dict.values(), 'description':cat_dict.keys()})
    if cl is None:
        warnings.warn('\n##-----## \nWarning: No category dictionary file was given, '\
                    'therefore the classes will be inferred from the material names. '\
                    'This might generate inconsistencies among images generated '\
                    'from different files. \n##-----##')
        classes_dict=None
    else:
        # labels_info = pd.read_csv(args.cl)
        # labels_info.description = labels_info.description.apply(lambda desc: unidecode.unidecode(desc).lower())
        classes_dict = utils.txt2dict(cl)
        classes_dict = {unidecode.unidecode(desc).lower(): i for desc,i in classes_dict.items()}

    # base numbering on the blender file
    start_ind = utils.last_file_index(
            glob.glob(os.path.join(blender_path, 'bridge*.blend'))) + 1

    for bridge in range(bridges):
        bridge_id = bridge+start_ind
        bl.clean_scene()

        ############################################################################
        #                LOAD AND INITIALISE MESH                                  #
        ############################################################################
        obj_file = os.path.join(obj_path, f'bridge{bridge_id}.obj')
        mtl_file = os.path.join(obj_path, f'bridge{bridge_id}.mtl')
        blender_file = os.path.join(blender_path, f'bridge{bridge_id}.blend')

        bridge_model = Bridge(json_data_path=basefile,
                obj_path=obj_file,
                label_path=mtl_file,
                json_param_path=params)
        bridge_model.create_mesh()
        bridge_model.save_mesh()
        # bridge_coords = {b.name: b.global_coords for b in bridge_model.building_blocks}
        del bridge_model
        bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='Y', axis_up='Z')
        obj = bpy.context.selected_objects[0] # returns an array of objects in the scene

        ############################################################################
        #                SPLIT TO SEMANTIC OBJECTS                                 #
        ############################################################################
        bl.bridgeObject2componentObjects(obj, classes_dict)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blender_file))


if __name__ == "__main__":
    args = get_args()
    generate_bridge_models(**vars(args))
    """
    ############################################################################
                                    END
    ############################################################################
    """
