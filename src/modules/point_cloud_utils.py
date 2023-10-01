import os
import sys
import pdb
import cv2
import glob
import json
import copy
import struct
import argparse
import datetime
import scipy.io
import numpy as np
import open3d as o3d
import numpy.typing as npt
from typing import Literal
from pygltflib import GLTF2
from scipy.spatial.transform import Rotation as R

from . import algebra_utils as alg
from .decorators import timer, verify_format


def data_from_accessor(glb, accessor):
    """
    return the data from an accessor to a numpy array
    """
    bufferView = glb.bufferViews[accessor.bufferView]
    buffer = glb.buffers[bufferView.buffer]
    data_binary = glb.get_data_from_buffer_uri(buffer.uri)
    # pull each vertex from the binary buffer and convert it into a tuple of python floats
    vertices = []
    for i in range(accessor.count):
        index = bufferView.byteOffset + accessor.byteOffset + i*12  # the location in the buffer of this vertex
        d = data_binary[index:index+12]  # the vertex data
        v = struct.unpack("<fff", d)   # convert from base64 to three floats
        vertices.append(v)
        # print(i, v)

    # unity uses a left-hand coordinate system, while blender uses a right hand one.
    # The GLB file uses the left hand convention (y axis is up). To transform it
    #to the right hand, a 90 degree rotation around the X axis is needed
    # return  np.asarray(vetrices)[:, [0,2,1]]
    vertices = alg.rotate_points(np.asarray(vertices).T, [90, 0, 0]).T
    return vertices


@verify_format('.glb')
def load_pointCloud_glb(filepath: str):
    """
    this function loads a glb file. Inspired from
    https://pypi.org/project/pygltflib/#reading-vertex-data-from-a-primitive-andor-getting-bounding-sphere
    see https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html#introduction
    for specifications
    """
    glb = GLTF2().load(filepath)
    # get the first mesh in the current scene (in this example there is only one scene and one mesh)
    mesh = glb.meshes[0]
    # get the vertices for each primitive in the mesh (in this example there is only one)
    # for primitive in mesh.primitives:

    # get the location binary data for this mesh primitive from the buffer
    accessor = glb.accessors[mesh.primitives[0].attributes.POSITION]
    data = data_from_accessor(glb, accessor)

    # get the color binary data, if they exist
    # if mesh.primitives[0].attributes.COLOR_0 is not None:
    #     accessor = glb.accessors[mesh.primitives[0].attributes.COLOR_0]
    #     colors = data_from_accessor(glb, accessor)
    #     data = np.concatenate((data, colors), axis=1)
    if 'labels' in mesh.primitives[0].extras.keys():
        labels = np.asarray(mesh.primitives[0].extras['labels'])
    else:
        labels = np.zeros((data.shape[0], 1))

    return data, labels


def intralabel_point_filter(points: npt.ArrayLike,
                            label: Literal['traverse', 'piedroit', 'mur', 'gousset', 'corniche'],
                            name: str):
    """
    this function chooses a subset of points, amongst all points of the same category
    that are appropriate to initialize an instance of this category
    ! It only works if the entire point cloud (whose subset is points) is centered
    around (0,0,0)
    """
    # TODO
    # MAYBE IMPROVE ABSOLUTES WITH GRADIENTS
    if label == 'traverse': # 1 candidate
        points = points[points[...,2]>0]
    elif label  in ['piedroit', 'gousset']:
        if '-W' in name:
            points = points[points[...,0]>0]
        elif '-E' in name:
            points = points[points[...,0]<0]
    elif label == 'mur':
        if '-NE' in name:
            points = points[points[...,0]<0]
            points = points[points[...,1]<0]
        elif '-NW' in name:
            points = points[points[...,0]>0]
            points = points[points[...,1]<0]
        elif '-SE' in name:
            points = points[points[...,0]<0]
            points = points[points[...,1]>0]
        elif '-SW' in name:
            points = points[points[...,0]>0]
            points = points[points[...,1]>0]
    elif label == 'corniche':
        if '-N' in name:
            points = points[points[...,1]<0]
        elif '-S' in name:
            points = points[points[...,1]>0]

    return points



""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
