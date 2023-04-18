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
import datetime
import numpy as np
from natsort import natsorted
from scipy import interpolate
# from numpy.polynomial import Polynomial
from mathutils import Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view
sys.path.append('../modules')
import algebra_utils
import blender_utils as bl


class Terrain:
    """
    This class represents the terrain, meaning the road
    and the ground surrounding the bridge
    """
    def __init__(self, bridge_coords, ground_width=20, subdivisions=5):
        coll = bpy.data.collections.new('terrain')
        bpy.context.scene.collection.children.link(coll)
        # dictionary with global coordinates for each building block
        self.bridge_coords = bridge_coords
        self.bridge_size = self.get_bridge_size()
        self.displacements = list()
        self.ground = dict()
        self.ground_width = max(ground_width, self.bridge_size[0]/2+5)
        self.road_poi = self.add_road()
        self.poi_pairs = self.get_POI_pairs()
        self.add_ground()
        bl.subdivide_meshes(*self.ground.values(), n_cuts=subdivisions)


    def get_bridge_size(self):
        all_coords = np.vstack(([block_coords for block_coords in self.bridge_coords.values()
                                            if block_coords is not None]))
        bridge_width = all_coords[:,0].max() - all_coords[:,0].min()
        bridge_length = all_coords[:,1].max() - all_coords[:,1].min()
        bridge_height = all_coords[:,2].max() - all_coords[:,2].min()

        return (bridge_width, bridge_length, bridge_height)


    def add_road(self):
        road_length = self.bridge_size[1] + 35
        # get coordinates of 4 edge points of bridge
        if 'piedroit-SW' in self.bridge_coords.keys() and 'piedroit-NW' in self.bridge_coords.keys():
            point_fl = self.bridge_coords['piedroit-SW'][7]
            point_bl = self.bridge_coords['piedroit-NW'][3]
        elif 'piedroit-W' in self.bridge_coords.keys():
            point_fl = self.bridge_coords['piedroit-W'][7]
            point_bl = self.bridge_coords['piedroit-W'][3]
        if 'piedroit-SE' in self.bridge_coords.keys() and 'piedroit-NE' in self.bridge_coords.keys():
            point_fr = self.bridge_coords['piedroit-SE'][6]
            point_br = self.bridge_coords['piedroit-NE'][2]
        elif 'piedroit-E' in self.bridge_coords.keys():
            point_fr = self.bridge_coords['piedroit-E'][6]
            point_br = self.bridge_coords['piedroit-E'][2]
        # extrapolate to get the edge road coordinates
        fr_ext, br_ext = algebra_utils.extend_3Dline_along_y(point_br, point_fr, road_length//2)
        fl_ext, bl_ext = algebra_utils.extend_3Dline_along_y(point_bl, point_fl, road_length//2)

        # add a plane to serve as the road
        bpy.ops.mesh.primitive_plane_add()
        bpy.data.objects['Plane'].name = 'road'
        bpy.data.meshes['Plane'].name = 'road'

        # stretch the road edges
        self.road = bpy.data.objects['road']
        bm = bmesh.new()
        bm.from_mesh(self.road.data)
        for vert in bm.verts:
            if vert.co == Vector((-1, -1, 0)):
                vert.co = br_ext
            elif vert.co == Vector((1, -1, 0)):
                vert.co = bl_ext
            elif vert.co == Vector((-1, 1, 0)):
                vert.co = fr_ext
            elif vert.co == Vector((1, 1, 0)):
                vert.co = fl_ext
        # Write back to the mesh
        bm.to_mesh(self.road.data)
        self.road.data.update()
        bm.clear()
        # put object in the collection corresponding to its class
        try:
            [x.objects.unlink(self.road) for x in self.road.users_collection]
            bpy.data.collections['terrain'].objects.link(self.road)
        except:
            pass

        poi = {'SE': fr_ext,
               'NE': br_ext,
               'SW': fl_ext,
                'NW': bl_ext,
                }

        return poi


    def add_ground(self, top_ground=None):
        # GROUND AROUND ROAD
        self.ground = dict.fromkeys(['east', 'west'])
        self.add_east()
        self.add_west()
        for side in self.ground.keys():
            try:
                [x.objects.unlink(self.ground[side]) for x in self.ground[side].users_collection]
                bpy.data.collections['terrain'].objects.link(self.ground[side])
            except:
                pass
        # if top_ground is not None:
        #     self.add_top(wing_walls, top_ground)


    def add_east(self):
        """
        adds a rough set of planes in order to fill the right side of the ground
        node numbering for hexaedra
        0: back top right
        1: back top left
        2: back bottom left
        3: back bottom right
        4: front top right
        5: front top left
        6: front bottom left
        7: front bottom right
        """
        vertices = np.empty((0,3))
        faces = []
        # 0-3 ROAD
        vertices = np.concatenate((vertices, self.road_poi['SE'][np.newaxis, ...]))
        vertices = np.concatenate((vertices, self.road_poi['SE'][np.newaxis, ...]
                                            + np.asarray([-self.ground_width, 0, 0])))
        vertices = np.concatenate((vertices, self.road_poi['NE'][np.newaxis, ...]))
        vertices = np.concatenate((vertices, self.road_poi['NE'][np.newaxis, ...]
                                             + np.asarray([-self.ground_width, 0, 0])))

        # 4-11 RIGHT ABUTMENT WALL (PIEDROIT)
        if 'piedroit-SE' in self.bridge_coords.keys() and 'piedroit-NE' in self.bridge_coords.keys():
            blockSE = self.bridge_coords['piedroit-SE']
            blockNE = self.bridge_coords['piedroit-NE']
        else:
            blockSE = blockNE = self.bridge_coords['piedroit-E']
        vertices = np.concatenate((vertices, blockNE[:4,...]))
        vertices = np.concatenate((vertices, blockSE[4:,...]))

        # some temporary coordinates needed if a wall is missing
        miss_wall_top_left = (vertices[4]+vertices[8])/2
        miss_wall_right = np.array([miss_wall_top_left[0]-random.uniform(self.ground_width/4-1, self.ground_width/4+1),
                                        miss_wall_top_left[1],
                                        miss_wall_top_left[2]-random.uniform(0, 2)])

        # 12-19 FRONT RIGHT WALL
        if 'mur-SE' not in self.bridge_coords.keys():
            #simulate coordinates based on exisiting elements
            simul = [miss_wall_right,
                    miss_wall_top_left,
                    vertices[10],
                    miss_wall_right]
            vertices = np.concatenate((vertices, simul))
            vertices = np.concatenate((vertices, simul))
            faces = [(12,13,14)]
        else:
            # take into account that the wing wall is rotated (90-180 deg ccw), its original
            # position being 90 degrees with the abutment wall facing west (positive x)
            # indices = np.array([5,4,7,6,1,0,3,2]) #4,0,3,7,5,1,2,6
            vertices = np.concatenate((vertices, self.bridge_coords['mur-SE'])) # self.bridge_coords['mur-SE'][indices,...]

        # 20-27 BACK RIGHT WALL
        if 'mur-NE' not in self.bridge_coords.keys():
            #simulate coordinates based on exisiting elements
            simul = [miss_wall_right,
                    miss_wall_top_left,
                    vertices[6],
                    miss_wall_right]
            vertices = np.concatenate((vertices, simul))
            vertices = np.concatenate((vertices, simul))
            faces.extend([(22,21,20)])
        else:
            # take into account that the wing wall is rotated, its original
            # position being 90 degrees with the abutment wall facing west (positive x)
            # indices = np.array([5,4,7,6,1,0,3,2])
            vertices = np.concatenate((vertices, self.bridge_coords['mur-NE']))

        assert len(vertices)==28, '{} vertices instead of 28!'.format(len(vertices))
        vertices = [tuple(x) for x in vertices]

        # define faces by sequence of counterclockwise vertices
        # (there are several possible combinations of faces to cover this space,
        # this is just one of them)
        faces.extend([(0,1,19), (0,18,10), (0,19,18), (1,12,19), (23,3,2), (22,23,2),
                    (3,23,24), (13,12,24), (13,24,25), (1,3,24,12), (2,6,22)])

        # Create Mesh Datablock
        mesh = bpy.data.meshes.new('ground_east')
        mesh.from_pydata(vertices, [], faces)
        # Create Object and link to scene
        self.ground['east'] = bpy.data.objects.new('ground_east', mesh)
        bpy.context.scene.collection.objects.link(self.ground['east'])
        bpy.context.view_layer.objects.active = self.ground['east']
        if bpy.context.mode == 'OBJECT':
            # Edit Mode
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0)
        bpy.ops.mesh.dissolve_degenerate()
        bpy.ops.object.mode_set(mode='OBJECT')


    def add_west(self):
        """
        adds a rough set of planes in order to fill the right side of the ground
        node numbering for hexaedra
        0: back top right
        1: back top left
        2: back bottom left
        3: back bottom right
        4: front top right
        5: front top left
        6: front bottom left
        7: front bottom right
        """
        vertices = np.empty((0,3))
        faces = []
        # 0-3 ROAD
        vertices = np.concatenate((vertices, self.road_poi['SW'][np.newaxis, ...]))
        vertices = np.concatenate((vertices, self.road_poi['SW'][np.newaxis, ...]
                                            + np.asarray([self.ground_width, 0, 0])))
        vertices = np.concatenate((vertices, self.road_poi['NW'][np.newaxis, ...]))
        vertices = np.concatenate((vertices, self.road_poi['NW'][np.newaxis, ...]
                                             + np.asarray([self.ground_width, 0, 0])))

        # 4-11 LEFT ABUTMENT WALL (PIEDROIT)
        if 'piedroit-SW' in self.bridge_coords.keys() and 'piedroit-NW' in self.bridge_coords.keys():
            blockSW = self.bridge_coords['piedroit-SW']
            blockNW = self.bridge_coords['piedroit-NW']
        else:
            blockSW = blockNW = self.bridge_coords['piedroit-W']
        vertices = np.concatenate((vertices, blockNW[:4,...]))
        vertices = np.concatenate((vertices, blockSW[4:,...]))

        # some temporary coordinates needed if a wall is missing
        miss_wall_top_right = (vertices[5]+vertices[9])/2
        miss_wall_left = np.array([miss_wall_top_right[0]+random.uniform(self.ground_width/4-1, self.ground_width/4+1),
                                        miss_wall_top_right[1],
                                        miss_wall_top_right[2]-random.uniform(0, 2)])

        # 12-19 FRONT LEFT WING WALL
        # if self.bridge_coords['mur-SW'] is None:
        if 'mur-SW' not in self.bridge_coords.keys():
            #simulate coordinates based on exisiting elements
            simul = [miss_wall_top_right,
                    miss_wall_left,
                    miss_wall_left,
                    vertices[11]]
            vertices = np.concatenate((vertices, simul))
            vertices = np.concatenate((vertices, simul))
            faces = [(13,15,12)]
        else:
            vertices = np.concatenate((vertices, self.bridge_coords['mur-SW']))

        # 20-27 BACK LEFT WING WALL
        # if self.bridge_coords['mur-NW'] is None:
        if 'mur-NW' not in self.bridge_coords.keys():
            #simulate coordinates based on exisiting elements
            simul = [miss_wall_top_right,
                    miss_wall_left,
                    miss_wall_left,
                    vertices[7]]
            vertices = np.concatenate((vertices, simul))
            vertices = np.concatenate((vertices, simul))
            faces.extend([(23,22,20)])
        else:
            vertices = np.concatenate((vertices, self.bridge_coords['mur-NW']))

            assert len(vertices)==28, '{} vertices instead of 28!'.format(len(vertices))
        vertices = [tuple(x) for x in vertices]

        # define faces by sequence of counterclockwise vertices
        # (there are several possible combinations of faces to cover this space,
        # this is just one of them)
        faces.extend([(0,11,19), (0,19,18), (0,18,1), (1,18,13), (13,12,24), (24,25,13),
                (22,23,2), (2,3,22), (3,25,22), (3,1,13,25), (2,23,7)])
        # faces.extend([(3,23,7)])

        # Create Mesh Datablock
        mesh = bpy.data.meshes.new('ground_west')
        mesh.from_pydata(vertices, [], faces)
        # Create Object and link to scene
        self.ground['west'] = bpy.data.objects.new('ground_west', mesh)
        bpy.context.scene.collection.objects.link(self.ground['west'])
        bpy.context.view_layer.objects.active = self.ground['west']
        if bpy.context.mode == 'OBJECT':
            # Edit Mode
            bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.remove_doubles(threshold=0)
        bpy.ops.mesh.dissolve_degenerate()
        bpy.ops.object.mode_set(mode='OBJECT')


    def get_POI_pairs(self):
        """
        save some points of interest that will later help to check collinearity
        of new points
        """
        point_pairs = list()
        # road and ground lines
        point_pairs.append((self.road_poi['SE'], self.road_poi['NE']))
        point_pairs.append((self.road_poi['SW'],self.road_poi['NW']))
        # outer limits of left ground
        point1 = self.road_poi['NW'] + np.asarray([self.ground_width, 0, 0])
        point2 = self.road_poi['SW'] + np.asarray([self.ground_width, 0, 0])
        point_pairs.append((point2, point1))
        point_pairs.append((point2,self.road_poi['SW']))
        point_pairs.append((point1,self.road_poi['NW']))
        # outer limits of right ground
        point1 = self.road_poi['NE'] - np.asarray([self.ground_width, 0, 0])
        point2 = self.road_poi['SE'] - np.asarray([self.ground_width, 0, 0])
        point_pairs.append((point2,point1))
        point_pairs.append((point2,self.road_poi['SE']))
        point_pairs.append((point1,self.road_poi['NE']))

        # WING WALLS
        for key in ['mur-NE', 'mur-SE', 'mur-NW', 'mur-SW']:
            if key in self.bridge_coords.keys():
                point_pairs.append((self.bridge_coords[key][1], self.bridge_coords[key][0]))
                point_pairs.append((self.bridge_coords[key][5], self.bridge_coords[key][4]))

        # ABUTMENT WALLS
        for key in ['piedroit-E', 'piedroit-W', 'piedroit-NE', 'piedroit-SE', 'piedroit-NW', 'piedroit-SW']:
            if key in self.bridge_coords.keys():
                point_pairs.append((self.bridge_coords[key][5], self.bridge_coords[key][1]))
                point_pairs.append((self.bridge_coords[key][4], self.bridge_coords[key][0]))

        return point_pairs


    def randomize_elevation(self, subdivisions=3, up_margin=0.6, tol=1):
        """
        this functions subdivides the ground plane in order to obtain a finer grid,
        and then lifts up(z-axis) some of the new vertices in order to get a
        randomized elevation profile
        INPUTS:
        @ground: ground object
        @subdivisions: (int) number of times to subdivide the grid
        @up_margin: (float) how many meters maximum can a point be elevated
        @tol: (float) how far from an original mesh coordinate a new vertex has to b
            in order to be considered for displacement
        """
        # define maximum height as the top of the abutment walls, since if the ground
        # goes above that it can lead to holes in the terrain
        max_height = list()
        for key in ['piedroit-E', 'piedroit-W', 'piedroit-NE', 'piedroit-SE', 'piedroit-NW', 'piedroit-SW']:
            if key in self.bridge_coords.keys():
                max_height.append(max([v[2] for v in self.bridge_coords[key]]))
        max_height = max(max_height)

        # remove vertices that are a bit too close to each other
        bl.remove_doubles_from_meshes(*self.ground.values(), dist=0.3)
        for side_desc, side in self.ground.items():
            rnd_gen = np.random.default_rng()
            for v in side.data.vertices:
                flag = False
                for poi_set in self.poi_pairs:
                    if algebra_utils.collinear(poi_set[0], poi_set[1], side.matrix_world@v.co, tol=tol):
                        flag = True
                        break
                if not flag:
                    elev = min(rnd_gen.uniform(0, up_margin), max_height - v.co[2]) # *random.choices([0, 1], weights=[0.8, 0.2])[0], if we want it more sparse
                    v.co[2] += elev
                    self.displacements.append((v, elev))


    def clear_displacements(self):
        """
        clear all previous randomized displacements and bring mesh to original state
        after subdivision but before randomization
        """
        for disp in self.displacements:
            disp[0].co[2] -= disp[1]
        self.displacements = list()
    """
    ############################################################################
    ############################################################################
    """
