"""
This scripts generates a synthetic dataset of bridge views along with their annotation
starting from a number of blender files
# sample usage remote: python generate_synthetic_data.py -input ../../data/Bridge_parametric_models/PIPO/blender/bridge*.blend -savefolder ../../data/Bridge_parametric_models/PIPO/ -textures ../../data/textures/bridge_materials/bridge_materials.blend
# sample usage local: python generate_synthetic_data.py -input ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/blender/bridge*.blend -savefolder ~/mounts/deepmachine/MIRAUAR/data/Bridge_parametric_models/PIPO/ -textures ~/mounts/deepmachine/MIRAUAR/data/textures/bridge_materials/bridge_materials.blend
"""
import os
import sys
import pdb
import bpy
import cv2
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
import pandas as pd
sys.path.append('modules')
import utils
import blender_utils as bl
import visualization as vis
import dataset_utils as dts
from decorators import forall
from refine_dataset import discard_unrelevant
from dataset_overview_and_stats import mist_demo, dataset_overview


# how many times to subdivide meshes so that the vertex visibility works better
n_cuts = 2


def restricted_float(x):
    """
    defines a float that is limited to be between 0 and 1
    """
    if x is None:
        return x
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} not a floating-point literal')
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError(f'{x} not in range [0, 1]')
    return x


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-input', help='wildcard for blender files', type=str, nargs='+',  required=True)
    arg('-savefolder', help= 'folder to save generated dataset',required=True, type=str)
    arg('-resx', help='X resolution (width) of output images [default: 640]', type=int, default=640)
    arg('-resy', help='Y resolution (height) of output images [default: 480]', type=int, default=480)
    arg('--struct-cov', help='minimum coverage, percentage of structure that should be in the frame for the image to be kept in the dataset [default:0.1]', type=restricted_float, default=0.1)
    arg('--img-cov', help='minimum coverage, percentage of image that contains the structure, for the image to be kept in the dataset [default:0.1]', type=restricted_float, default=0.1)
    arg('-frames', help='number of camera angle views per bridge [default: 1]', type=int, default=1)
    arg('-textures', help='path to folder with texture files', type=str)
    # arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)
    args = parser.parse_args()
    return args


@forall
def render_labeled_images(*args, **kwargs):
    """
    INPUTS:
    @args: for the moment one positional argument, the path to the
            blender file
    """
    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(args[0]))
    class_dict = bl.infer_class_dict()
    kwargs.update({'class_dict': class_dict})
    bl.initialize_blender_env(**kwargs)
    scene = bpy.context.scene

    ############################################################################
    #                             CAMERA                                       #
    ############################################################################
    # add camera to the scene
    camera = bl.add_camera()

    ############################################################################
    #                              TEXTURES                                    #
    ############################################################################
    # SET TEXTURES
    ground_materials = [mat for mat in bpy.data.materials
                                if mat.asset_data.catalog_simple_name == "ground"]
    ground_objs = [obj for obj in bpy.data.objects if 'ground' in obj.name]
    bl.set_materials(ground_materials, ground_objs)
    road_materials = [mat for mat in bpy.data.materials
                                if mat.asset_data.catalog_simple_name == "road"]
    road_objs = [obj for obj in bpy.data.objects if 'road' in obj.name]
    bl.set_materials(road_materials, road_objs)
    wall_materials = [mat for mat in bpy.data.materials
                                if mat.asset_data.catalog_simple_name == "walls"]
    bridge_objs = [obj for obj in bpy.data.objects if
                    obj.users_collection[0].name in class_dict.keys()]
    bl.set_materials(wall_materials, bridge_objs)

    ############################################################################
    #                          MESH OPERATIONS                                #
    ############################################################################
    # enable smooth shading and auto smooth for better render experience
    for mesh in bpy.data.meshes:
        for pol in mesh.polygons:
            pol.use_smooth=True
            # mesh.use_auto_smooth=True

    # subdivide meshes so that the vertex visibility works better
    bl.subdivide_meshes(*bridge_objs, n_cuts=n_cuts)

    ############################################################################
    #               INSTANTIATE SETUP AND RENDER                               #
    ############################################################################
    # SET SKY PARAMETERS
    bl.set_sky(scene)
    # COMPOSITING GRAPH OUTPUTS
    bl.set_compositing_graph(scene=scene,
        bridgename=os.path.splitext(os.path.basename(args[0]))[0])

    #  RENDER
    # generate a number of images with randomly sampled camera locations
    # and rotations, for a specific number of cuts (defining a number of vertices)
    bl.generate_renders(objects=bridge_objs,
                    n_frames=kwargs['frames'],
                    min_coverage=kwargs['struct_cov'])

    # monitor the files that are fully rendered
    with open(os.path.join(os.path.dirname(args[0]), 'rendered.txt'), 'a') as f:
        f.write(f'#-------------#\n{datetime.datetime.now}\n#-------------#\n')
        f.write(f'{os.path.basename(args[0])}\n\n')


if __name__ == "__main__":
    nl='\n';
    args = get_args()
    paths = dict()
    # dts_id = utils.dataset_id(args.savefolder)
    dts_id = 'dataset_MIRAUAR'
    paths['base'] = utils.create_folders(args.savefolder, dts_id)[0]
    paths['demo'], paths['overview'] = utils.create_folders(paths['base'],
                                                        'demo', 'overview')
    paths['info'] = os.path.join(paths['base'], 'info.txt')

    # write some information regarding the experiment in the info file
    with open(paths['info'], 'a') as f:
        f.write(f'Input files: {nl} {nl.join(args.input)} \n\n')
        f.write(f'# images per bridge: {args.frames} \n\n')
        f.write(f'Image resolution: {args.resx} x {args.resy} \n\n')
        f.write(f'at least {args.struct_cov*100}% of bridge should be in the image \n\n')
        f.write(f'at least {args.img_cov*100}% of image should be consisted of bridge \n\n')

    # input files are passed as args, rest values are passed as kwargs
    kwargs = {key: val for key, val in vars(args).items() if key!='input'}
    kwargs['savefolder'] = paths['base']
    render_labeled_images(*args.input, **kwargs)

    # save info on what device was used for render
    with open(paths['info'], 'a') as f:
        f.write('Device usage: \n')
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            is_used = {0:"not used", 1: "used"}[d.use]
            f.write(f'{d.name}: {is_used}\n')
        f.write('\n\n')

    ############################################################################
    #                             POST PROCESS                                 #
    ############################################################################
    discard_unrelevant(paths['base'], args.img_cov)

    image_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(paths['base'], 'images'))
    mask_paths = utils.files_with_extensions('jpg', 'JPG', 'png',
                    datapath=os.path.join(paths['base'], 'masks', '**'), recursive=True)
    # in case there is no image, delete folder
    if len(image_paths)==0:
        print(f'Not a single acceptable image was produced, dataset {args.d} is removed.\n')
        shutil.rmtree(paths['base'])
        sys.exit()

    # demo of mist no mist
    mist_demo(image_paths, paths['demo'], clear=True)
    # dataset overview
    dataset_overview(image_paths, mask_paths, paths['overview'])
    """
    ############################################################################
                                    END
    ############################################################################
    """
