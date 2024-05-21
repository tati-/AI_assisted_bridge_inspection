import os
import sys
import json
import math
import random
import flatdict
import numpy as np
import open3d as o3d
import numpy.typing as npt
from typing import Literal
from dataclasses import dataclass
from scipy.spatial import ConvexHull

import modules.algebra_utils as alg


class Hexahedron:
    """
    this class represents a geometric element, an hexahedron used as a building
    block for the bridge
    """
    def __init__(self,
                id: int,
                label: str,
                name: str,
                width: npt.ArrayLike,
                length: npt.ArrayLike,
                height: npt.ArrayLike,
                rotx: float=0,
                roty: float=0,
                rotz: float=0):
        """
        @json_data: the list of dictionary items read from the diades json file
                    (check readme code for its structure)
        @self.points: nPoints x 3 array containing the points that belong to a block
        @self.plane: coefficients of plane equation describing a block,
                    array of the form [a,b,c,d] where
                    ax + by + cz + d = 0 is the plane equation
        """
        # this id is the index of the current block in the blocks array
        # can I disassociate this?
        self.id = id
        self.label = label
        self.name = name
        # width, length, height
        self.dimensions = dict.fromkeys(['width', 'length', 'height'])
        # x, y, z axis rotation angles in degrees
        self.rotations = dict.fromkeys(['roll', 'tilt', 'heading'])
        self.set_dimensions(width, length, height)
        self.set_rotations(rotx, roty, rotz)
        self.offset = np.zeros(3)
        self.visible = True
        self.points, self.plane = None, None


    def from_points(self,
                    points: npt.ArrayLike,
                    box_type: Literal['oriented', 'axis_aligned']='oriented'):
        """
        this function rougly defines a block from a set of points that should
        be enclosed by it
        """
        self.points = points
        # limits = np.max(points, axis=0) - np.min(points, axis=0)
        # axis_order = np.argsort(limits)[::-1] # indices that sort limits in descending order
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:,:3])
        # if points.shape[1]>3:
        #     pcd.colors = o3d.utility.Vector3dVector(points[:,3:6]) # should be in [0,1] /255
        #     if points.shape[1]>6:
        #         pcd.normals = o3d.utility.Vector3dVector(point_cloud[:,6:9])
        # o3d.visualization.draw_geometries([pcd])
        if box_type=='axis_aligned':
            bbox = pcd.get_axis_aligned_bounding_box()
        elif box_type=='oriented':
            bbox = pcd.get_minimal_oriented_bounding_box()
        # box_points = np.asarray(bbox.get_box_points())
        # box_center = np.asarray(bbox.center)
        # # value range per axis - bbox.extent is ordered in descending order of
        # # explained variance pca component
        # box_points = R.from_matrix(rot_matrix).apply(box_points)
        # dimensions =  bbox.extent # [axis_order] #np.max(box_points, axis=0) - np.min(box_points, axis=0)
        dimensions = np.ptp(points, axis=0)
        # rot_matrix = copy.deepcopy(bbox.R)
        # rot_angles = R.from_matrix(rot_matrix).as_euler('xyz', degrees=True)
        # infer initial angles and tilt - TODO
        # # width - TODO, points need to be rotated for this to work
        # D1 = box_points[4,0] - box_points[5,0]
        # D2 = box_points[7,0] - box_points[2,0]
        #
        width = [dimensions[0], dimensions[0], 0] # D1, D2, tilt
        length = [dimensions[1], dimensions[1], 0]
        height = [dimensions[2], dimensions[2], 0]
        # initialize block
        # bbox.color = (1, 0, 0)
        # o3d.visualization.draw_geometries([pcd, bbox])
        self.set_dimensions(width, length, height)
        self.set_rotations(0,0,0)


    def set_dimensions(self,
                        width: npt.ArrayLike,
                        length: npt.ArrayLike,
                        height: npt.ArrayLike):
        """
        This function sets the dimensions of the cube and the tilt(decalage)
        INPUTS:
        @width: length-3 array, first element: top cube width
                               second element: bottom cube width
                               third element: width offset
        @length: length-3 array, first element: left cube length
                               second element: right cube length
                               third element: length offset
        @height: length-3 array, first element: left cube height
                               second element: right cube height
                               third element: height offset
        OUTPUTS:
        dimension: string, indication the dimension, in {'Longueur', 'Largeur', 'Hauteur'}
        dim1: first dimension (towards positive face)
        dim2: second dimension (towards negative face)
        tilt: offset between the 2 parallel lines, allowing the cube to tilt
        """
        self.dimensions['width'] = np.asarray(width)
        self.dimensions['length'] = np.asarray(length)
        self.dimensions['height'] = np.asarray(height)


    def set_rotations(self,
                    rotx: float,
                    roty: float,
                    rotz: float):
        """
        This function sets the rotation angle in degrees
        rot_axis: string, indication of the rotation axis
        If the obj is loaded with Y forward, Z up, this corresponds to:
                z: 'Heading' (bridge height - hauteur)
                y: 'Tilt' (bridge length - largeur)
                x: 'Roll' (bridge width - longueur)
        INPUTS:
        @rotx, roty, rotz: rotation angles in degrees
        """
        self.rotations['roll'] = rotx
        self.rotations['tilt'] = roty
        self.rotations['heading'] = rotz


    @property
    def volume(self):
        """
        calculate the hexahedron's volume
        """
        return alg.volume_from_points(self.local_coords)


    def get_local_node_coords(self, node_id: int) -> npt.ArrayLike:
        """
        given a node (vertex) id, return
        the node 3d coordinates (assuming the cube center is at (0,0,0)
        INPUTS:
        node_id: integer in {0,...,7}, denoting the vertex
        OUTPUTS:
        3d coordinates of node

        NOTE: the node positioning comments refer to a y-forward, z-up view
        @width: length-3 array, first element: top cube width
                               second element: bottom cube width
                               third element: width tilt
        @length: length-3 array, first element: left cube length
                               second element: right cube length
                               third element: length tilt
        @height: length-3 array, first element: left cube height
                               second element: right cube height
                               third element: height tilt
        """
        # the d1 and d2 (first two element) of the dimensions refer to the overal
        # hexahedron size, so we divide them by two, to get the coordinates
        width = self.dimensions['width']/np.array([2,2,1])
        length = self.dimensions['length']/np.array([2,2,1])
        height = self.dimensions['height']/np.array([2,2,1])
        if node_id==0: # top back right
            coords = [-width[0], -length[1]+length[2], height[1]+height[2]]
        elif node_id==1: # top back left
            coords = [width[0], -length[0], height[0]]
        elif node_id==2: # bottom back left
            coords = [width[1]+width[2], -length[0], -height[0]]
        elif node_id==3: # bottom back right
            coords = [-width[1]+width[2], -length[1]+length[2], -height[1]+height[2]]
        elif node_id==4: # top front right
            coords = [-width[0],length[1]+length[2], height[1]+height[2]]
        elif node_id==5: # top front left
            coords = [width[0], length[0], height[0]]
        elif node_id==6: # bottom front left
            coords = [width[1]+width[2], length[0], -height[0]]
        elif node_id==7: # bottom front right
            coords = [-width[1]+width[2], length[1]+length[2], -height[1]+height[2]]
        else:
            sys.exit(f'Error: node_id must be an integer between 0 and 7, is {node_id} instead')
        return np.asarray(coords)


    @property
    def local_coords(self):
        """
        This function calculates and returns the local coordinates ([0,0,0] is the
        center of the hexahedron) of all the nodes of the hexahedron
        # top back right      - 0
        # top back left       - 1
        # bottom back left    - 2
        # bottom back right   - 3
        # top front right     - 4
        # top front left      - 5
        # bottom front left   - 6
        # bottom front right  - 7
        @self.local_coords: 8x3 array, global hexahedron coordinates after rotation
        """
        rot_angles = [self.rotations['roll'], # x axis
                      self.rotations['tilt'], # y axis
                      self.rotations['heading'] # z axis
                     ]

        # for an incomplete hexahedron (where some dimensions are None),
        # local coordinates will all be reduced to 0
        if self.is_complete():
            # get local node coordinates before rotation
            local_coords = np.asarray([self.get_local_node_coords(node) for node in range(8)])

            local_coords = alg.rotate_points(coord=local_coords.T, rot_angles=rot_angles).T
        else:
            local_coords = np.zeros((8,3))

        return local_coords


    @property
    def global_coords(self):
        """
        gets the global coords taking into account the local coords and the offset
        """
        return self.local_coords + self.offset


    @property
    def rect_faces(self):
        """
        returns a np array of shape 6x4x3, where each line contains the vertex
        coordinates that compose a face in counterclockwise order
        0: front
        1: back
        2: top
        3: bottom
        4: right
        5: left
        """
        faces = np.array([[6, 7, 4, 5],
                          [1, 0, 3, 2],
                          [5, 4, 0, 1],
                          [2, 3, 7, 6],
                          [4, 7, 3, 0],
                          [6, 5, 1, 2]])
        return faces
        # faces = [np.array([self.global_coords[v] for v in face]) for face in faces_ind]
        # return np.stack(faces, axis=0)


    @property
    def triangle_mesh(self):
        """
        returns a np array of shape 6x4x3, where each line contains the vertex
        coordinates that compose a triangular face in counterclockwise order
        0, 1: front
        2, 3: back
        4, 5: top
        6, 7: bottom
        8, 9: right
        10, 11: left
        """
        faces = np.array([[6, 7, 4],
                          [4, 5, 6],
                          [1, 0, 3],
                          [3, 2, 1],
                          [5, 4, 0],
                          [0, 1, 5],
                          [2, 3, 7],
                          [7, 6, 2],
                          [4, 7, 3],
                          [3, 0, 4],
                          [6, 5, 1],
                          [1, 2, 6]])

        vertices = o3d.utility.Vector3dVector(self.global_coords)
        triangles = o3d.utility.Vector3iVector(faces)
        trimesh = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
        return trimesh
        # faces = [np.array([self.global_coords[v] for v in face]) for face in faces_ind]
        # return np.stack(faces, axis=0)


    def is_complete(self) -> bool:
        """
        check if all of the hexahedron dimensions and rotations
        are filled with a value (not None)
        """
        missing_val = [x is None for x in np.concatenate(flatdict.FlatDict(self.dimensions, delimiter='.').values())]
        missing_val.extend([x is None for x in flatdict.FlatDict(self.rotations, delimiter='.').values()])
        if np.any(missing_val):
            return False
        else:
            return True


    def contains(self, points: npt.ArrayLike) -> npt.ArrayLike:
        """
        checks if points lie inside the hexahedron.
        Returns a boolean array, with a value for each one of the points
        INPUTS:
        points: nPoints x 3 array with the points 3d coordinates
        """
        return alg.points_in_mesh(self.global_coords, points)


    def to_dict(self) -> dict:
        """
        creates a dictionary out of the hexahedron(complying with the json file
        rules). Since this is a single blocks, no constraints are assumed
        """
        # initialize dictionary fields
        hex_dict = dict.fromkeys(['index', 'label', 'name', 'dimensions', 'angle', 'contstraint'])
        hex_dict['dimensions'] = {key: dict.fromkeys(['D1', 'D2', 'tilt', 'constraint', 'tilt_constraints'])
                                    for key in ['width', 'length', 'height']}
        hex_dict['angles'] = {key: dict.fromkeys('V', 'constraint') for key in ['heading', 'tilt', 'roll']}
        hex_dict['constraint'] = dict.fromkeys(['parent', 'parent_node', 'child_node'], -1)
        # initialize all conrainte values to -1, since they are only relevant for a list of
        # hexahedra
        for key, val in hex_dict['dimensions'].items():
            val['constraint'] = -1
            val['offset_constraint'] = -1
        for key, val in hex_dict['angles'].items():
            val['constraint'] = -1
        # fill values with object attributes
        hex_dict['index'] = self.id
        hex_dict['label'] = self.label
        hex_dict['name'] = self.name
        for key in hex_dict['dimensions'].keys():
            hex_dict['dimensions'][key]['D1'] = self.dimensions[key][0]
            hex_dict['dimensions'][key]['D2'] = self.dimensions[key][1]
            hex_dict['dimensions'][key]['tilt'] = self.dimensions[key][2]
        for key in hex_dict['angles'].keys():
            hex_dict['angles'][key]['V'] = self.rotations[key]

        return hex_dict


    def to_json(self, savepath: str):
        """
        saves the block in json format
        """
        # create and save json file
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        with open(savepath, 'w') as fp:
            json.dump([self.to_dict()], fp)


    @property
    def fv(self) -> npt.ArrayLike:
        """
        Convert hexahedron dimensions to a feature vector to be used in
        optimization algorithms
        """
        fv = np.zeros(12)
        fv[0:3] = self.dimensions['width']
        fv[3:6] = self.dimensions['length']
        fv[6:9] = self.dimensions['height']
        fv[9] = self.rotations['roll']
        fv[10] = self.rotations['tilt']
        fv[11] = self.rotations['heading']

        return fv


    def from_fv(self, fv: npt.ArrayLike):
        """
        Convert fv to hexahedron dimensions
        INPUTS:
        @fv: array of length 12, with the hexahedron dimensions in the
            implied order
        """
        self.dimensions['width'] = fv[0:3]
        self.dimensions['length'] = fv[3:6]
        self.dimensions['height'] = fv[6:9]
        self.rotations['roll'] = fv[9]
        self.rotations['tilt'] = fv[10]
        self.rotations['heading'] = fv[11]

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

@dataclass
class Constraint:
    """
    This class represents a single constraint
    """
    width: int=-1
    length: int=-1
    height: int=-1
    offset_width: int=-1
    offset_length: int=-1
    offset_height: int=-1
    roll: int=-1
    tilt: int=-1
    heading: int=-1
    parent: int=-1
    parent_node: int=-1
    child_node: int=-1
