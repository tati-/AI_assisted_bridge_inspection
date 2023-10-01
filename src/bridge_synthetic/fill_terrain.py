"""
This scripts gets a number of blender files representing bridges and fill the
terrain around them.
# sample usage: python fill_terrain.py -input ../../data/PIPO/3d_models/blender/bridge*.blend [bridge{0..10}.blend]
"""
import os
import re
import sys
import pdb
import bpy
import glob
import math
import time
import random
import argparse
import warnings
import datetime
import numpy as np

sys.path.append('..')
import modules.blender_utils as bl
from modules.decorators import forall
from classes.terrain import Terrain

@forall
def fill_terrain(filename):
    """
    gets a number of blender files representing a bridge, adds a road below
    it and fills the terrain around it
    """
    if os.path.splitext(filename)[1]!='.blend': return
    bl.clean_scene()
    #import bridge
    bpy.ops.wm.open_mainfile(filepath=filename)
    # remove road and ground objects if they exist already
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects
                            if obj.type=='MESH' and
                            re.search('road|ground_east|ground_west', obj.name, flags=re.I)]
    [bpy.data.collections.remove(col) for col in bpy.data.collections if
                                        'terrain' in col.name]
    # get a dictionary with the coordinates of each bridge component as
    # a hexahedron, to use as anchor points for the terrain filling
    bridge_coords = {obj.name:bl.ojb2hexahedronCoords(obj) for obj in bpy.data.objects
                            if obj.type=='MESH'}
    terrain = Terrain(bridge_coords=bridge_coords)
    terrain.randomize_elevation()  # randomize a bit elevation profile
    # save the updated file
    bpy.ops.wm.save_as_mainfile() # , visible_objects_only=True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-input', help='wildcard for blender files', type=str, nargs='+',  required=True)
    args = parser.parse_args()

    fill_terrain(*args.input)
    """
    ############################################################################
                                    END
    ############################################################################
    """
