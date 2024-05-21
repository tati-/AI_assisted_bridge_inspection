import os
import sys
import math
import struct
import random
import numpy as np
import open3d as o3d
import numpy.typing as npt
from typing import Literal
from pygltflib import GLTF2
from collections import Counter
from scipy.spatial.transform import Rotation as R

from . import algebra_utils as alg
from .constants import CLASSES_PIPO
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
    data = np.empty((0, 3))
    labels = np.empty((0), dtype=int)

    for mesh in glb.meshes:
        # get the vertices for each primitive in the mesh (in this example there is only one)
        # for primitive in mesh.primitives:

        # get the location binary data for this mesh primitive from the buffer
        accessor = glb.accessors[mesh.primitives[0].attributes.POSITION]
        data_tmp = data_from_accessor(glb, accessor)
        data = np.append(data, data_tmp, axis=0)
        # get the color binary data, if they exist
        # if mesh.primitives[0].attributes.COLOR_0 is not None:
        #     accessor = glb.accessors[mesh.primitives[0].attributes.COLOR_0]
        #     colors = data_from_accessor(glb, accessor)
        #     data = np.concatenate((data, colors), axis=1)
        if 'labels' in mesh.primitives[0].extras.keys():
            labels = np.append(labels, np.asarray(mesh.primitives[0].extras['labels']), axis=0)
        else:
            try:
                label = [val for key, val in CLASSES_PIPO.items() if key in mesh.name][0]
            except:
                label = 0
            labels = np.append(labels, np.full((len(data_tmp)), fill_value=label), axis=0)

    return data, labels


def intralabel_point_filter(points: npt.ArrayLike,
                            label: Literal['deck', 'abutment', 'wing_wall', 'haunch', 'edge_beam'],
                            name: str):
    """
    this function chooses a subset of points, amongst all points of the same category
    that are appropriate to initialize an instance of this category
    """
    # TODO
    # MAYBE IMPROVE ABSOLUTES WITH GRADIENTS
    mean_point = np.mean(points, axis=0)
    # if label == 'traverse': # 1 candidate
    #     points = points[points[...,2]>mean_point[2]]
    if label  in ['abutment', 'haunch']: #['piedroit', 'gousset']:
        if '-w' in name:
            points = points[points[...,0]>mean_point[0]]
        elif '-e' in name:
            points = points[points[...,0]<mean_point[0]]
    elif label == 'wing_wall': #'mur':
        if '-ne' in name:
            points = points[points[...,0]<mean_point[0]]
            points = points[points[...,1]<mean_point[1]]
        elif '-nw' in name:
            points = points[points[...,0]>mean_point[0]]
            points = points[points[...,1]<mean_point[1]]
        elif '-se' in name:
            points = points[points[...,0]<mean_point[0]]
            points = points[points[...,1]>mean_point[1]]
        elif '-sw' in name:
            points = points[points[...,0]>mean_point[0]]
            points = points[points[...,1]>mean_point[1]]
    elif label == 'edge_beam': #'corniche':
        if '-n' in name:
            points = points[points[...,1]<mean_point[1]]
        elif '-s' in name:
            points = points[points[...,1]>mean_point[1]]

    return points


@timer
def majority_filter(points: npt.ArrayLike,
                    labels: npt.ArrayLike,
                    radius: float=0.15) -> npt.ArrayLike:
    """
    filters the labels of a point cloud so each point's label is sunstituded by the
    most frequent label of its neighbors of a certain radius.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)

    for i, point in enumerate(points):
        # find points that belong in radius k: point count, idx: list of point indices
        k, idx, _ = pcd_tree.search_radius_vector_3d(point, radius)

        # find majority class
        maj_class = Counter(labels[idx]).most_common()[0][0]
        # values, counts = np.unique(labels[idx], return_counts=True)
        # maj_class = values[np.argmax(counts)]

        # substitute point label with the majority class
        if maj_class!=0:
            labels[i] = maj_class

    return labels


def add_noise(labels: npt.ArrayLike,
            noise: float=0.0) -> npt.ArrayLike:
    """
    adds noise to the labels of a point cloud
    """
    if noise==0:
        return labels

    labels_present = list(set(labels))
    n_points = math.floor(noise*len(labels))
    indices_to_change = np.random.choice(range(len(labels)), n_points, replace=True)
    for i in indices_to_change:
        labels[i] = random.choice([x for x in labels_present if x!=labels[i]])
    return labels


def points2plane(points: npt.ArrayLike):
    """
    this function finds a plane from a set of points
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.13,
                                         ransac_n=7,
                                         num_iterations=1000)
    # [a, b, c, d] = plane_model
    # print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    return plane_model, inliers
    # inlier_cloud = pcd.select_by_index(inliers)
    # inlier_cloud.paint_uniform_color([0, 0, 0])
    # outlier_cloud = pcd.select_by_index(inliers, invert=True)
    # o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
    #                               zoom=0.8,
    #                               front=[-0.4999, -0.1659, -0.8499],
    #                               lookat=[2.1813, 2.0619, 2.0999],
    #                               up=[0.1204, -0.9852, 0.1215])


@timer
def plane_filter(points: npt.ArrayLike, labels: npt.ArrayLike) -> tuple:
    """
    filters a point cloud so that points belong to planes (discard outliers)
    segments plane with open3d, to eliminate outliers (RANSAC)
    INPUTS:
    @points: npoints x 3 array with 3d coordinates
    @labels : npoints length array with label per point
    OUTPUTS:
    same as inputs but filtered to only contain inliers
    """
    clean_points = np.empty((0,3))
    clean_labels = np.empty(0)
    for label in np.unique(labels):
        class_points = points[labels==label, :]
        if label==1: # abutment, one right (positive x) one left (negative x)
            pos_points = class_points[class_points[..., 0]>=0, ...]
            neg_points = class_points[class_points[..., 0]<0, ...]
            for subset_points in (pos_points, neg_points):
                _, inliers = points2plane(subset_points)
                clean_points = np.vstack((clean_points, subset_points[inliers, ...]))
                clean_labels = np.append(clean_labels, np.repeat(label, len(inliers)))
        elif label==2: #deck
            class_points = class_points[class_points[..., 2]>0, ...]
            _, inliers = points2plane(class_points)
            clean_points = np.vstack((clean_points, class_points[inliers, ...]))
            clean_labels = np.append(clean_labels, np.repeat(label, len(inliers)))
        elif label==3: # wing walls
            conditions = list()
            conditions.append(np.logical_and(class_points[..., 0]<0,
                                             class_points[..., 1]<0))
            conditions.append(np.logical_and(class_points[..., 0]<0,
                                             class_points[..., 1]>=0))
            conditions.append(np.logical_and(class_points[..., 0]>=0,
                                             class_points[..., 1]<0))
            conditions.append(np.logical_and(class_points[..., 0]>=0,
                                             class_points[..., 1]>=0))
            for subset_points in [class_points[cond, ...] for cond in conditions]:
                _, inliers = points2plane(subset_points)
                clean_points = np.vstack((clean_points, subset_points[inliers, ...]))
                clean_labels = np.append(clean_labels, np.repeat(label, len(inliers)))
        elif label==4: # haunch, one right (positive x) one left (negative x)
            conditions = list()
            conditions.append(np.logical_and(class_points[..., 0]>=0,
                                             class_points[..., 2]>0))
            conditions.append(np.logical_and(class_points[..., 0]<0,
                                             class_points[..., 2]>0))
            for subset_points in [class_points[cond, ...] for cond in conditions]:
                _, inliers = points2plane(subset_points)
                clean_points = np.vstack((clean_points, subset_points[inliers, ...]))
                clean_labels = np.append(clean_labels, np.repeat(label, len(inliers)))
        elif label==5: # edge beams, one front (positive y) one back (negative y)
            conditions = list()
            conditions.append(np.logical_and(class_points[..., 1]>=0,
                                             class_points[..., 2]>0))
            conditions.append(np.logical_and(class_points[..., 1]<0,
                                             class_points[..., 2]>0))
            for subset_points in [class_points[cond, ...] for cond in conditions]:
                _, inliers = points2plane(subset_points)
                clean_points = np.vstack((clean_points, subset_points[inliers, ...]))
                clean_labels = np.append(clean_labels, np.repeat(label, len(inliers)))
        else:
            clean_points = np.vstack((clean_points, class_points))
            clean_labels = np.append(clean_labels, np.repeat(label, class_points.shape[0]))
    return clean_points, clean_labels
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
