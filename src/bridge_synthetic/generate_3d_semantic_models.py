"""
This scripts generates a number of bridge 3d models along with their annotation
starting from a .json file with all necessary parameters
# sample usage: python generate_3d_semantic_models.py @3d_config.txt
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

sys.path.append('..')
import modules.constants as constants
import modules.utils as utils
import modules.blender_utils as bl
from modules.constants import CLASSES_PIPO
from classes.bridge import BridgeModel as Bridge
from modules.decorators import timer


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-basefile', help='input file path [acceptable formats: .json]', type=str, required=True)
    arg('-params', help='json file path with parameters', type=str)
    # arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)
    arg('-savefolder', help= 'folder to save generated dataset [default: same as input file directory]', type=str)
    arg('-bridges', help='number of bridges [default: 1]', type=int, default=1)
    args = parser.parse_args()
    return args


@timer
def generate_bridge_models(basefile, params=None, savefolder=None, cl=None, bridges=1):
    """
    Creates a number of annotated bridge 3d models and saves them in OBJ and
    blender format. If params argument is passed, the bridge dimensions are
    randomly selected for each one of the bridge models.
    """
    assert os.path.splitext(basefile)[1]=='.json', 'JSON format should be given as basefile'
    savefolder = os.path.dirname(basefile) if savefolder is None else savefolder
    obj_path, blender_path = utils.create_folders(savefolder, 'obj', 'blender')

    # labels_info = pd.DataFrame(data={'id': cat_dict.values(), 'description':cat_dict.keys()})
    # if cl is None:
    #     warnings.warn('\n##-----## \nWarning: No category dictionary file was given, '\
    #                 'therefore the classes will be inferred from the material names. '\
    #                 'This might generate inconsistencies among images generated '\
    #                 'from different files. \n##-----##')
    #     classes_dict=None
    # else:
    #     # labels_info = pd.read_csv(args.cl)
    #     # labels_info.description = labels_info.description.apply(lambda desc: unidecode.unidecode(desc).lower())
    #     classes_dict = utils.txt2dict(cl)
    #     classes_dict = {unidecode.unidecode(desc).lower(): int(i) for desc,i in classes_dict.items()}
    classes_dict = constants.CLASSES_PIPO

    # base numbering on the blender file
    start_ind = utils.last_file_index(
            glob.glob(os.path.join(blender_path, 'bridge*.blend'))) + 1

    for bridge in range(bridges):
        bridge_id = bridge+start_ind
        bl.clean_scene()

        ############################################################################
        #                LOAD AND INITIALISE MESH                                  #
        ############################################################################
        blender_file = os.path.join(blender_path, f'bridge{bridge_id}.blend')

        bridge_model = Bridge(json_data_path=basefile,
                json_param_path=params)
        bridge_model.to_blender(blenderPath=blender_file)
        # the BridgeModel object here only serves to create the 3d mesh
        del bridge_model

        # bpy.ops.import_scene.obj(filepath=obj_file, axis_forward='Y', axis_up='Z')
        # obj = bpy.context.selected_objects[0] # returns an array of objects in the scene
        #
        # ############################################################################
        # #                SPLIT TO SEMANTIC OBJECTS                                 #
        # ############################################################################
        # bl.bridgeObject2componentObjects(obj, classes_dict)
        # bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blender_file))


if __name__ == "__main__":
    args = get_args()
    generate_bridge_models(**vars(args))
    """
    ############################################################################
                                    END
    ############################################################################
    """
