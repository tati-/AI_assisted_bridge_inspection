import os
import re
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
import unidecode
import numpy as np
from pathlib import Path
from natsort import natsorted
from mathutils import Vector, Quaternion
from bpy_extras.object_utils import world_to_camera_view

from . import algebra_utils
from . import visualization as vis
from .decorators import iterate, optional, forall, verify_format


def clean_scene():
    # remove all objects and meshes from blender scene
    [bpy.data.objects.remove(obj) for obj in bpy.data.objects]
    [bpy.data.meshes.remove(mesh) for mesh in bpy.data.meshes]


def initialize_blender_env(**kwargs):
    """
    this function sets some general blender parameters in order to initialize
    the blender settings that will be used for the synthetic dataset
    generation
    """
    # set render parameters
    scene = bpy.context.scene # = bpy.data.scenes['Scene']
    scene.render.resolution_x = kwargs['resx']
    scene.render.resolution_y = kwargs['resy']
    scene.render.engine = 'CYCLES'
    scene.cycles.samples =  2048 # 1024 max samples to use for rendering
    # scene.cycles.texture_limit_render = '1024' # max image size to use for rendering
    bpy.data.scenes['Scene'].cycles.device = 'GPU'

    # Attempt to set GPU device types if available
    for compute_device_type in ('CUDA', 'METAL', 'OPENCL', 'NONE'):
    # for compute_device_type in ('CUDA', 'OPENCL', 'NONE'):
        try:
            bpy.context.preferences.addons['cycles'].preferences.compute_device_type = compute_device_type
            break
        except TypeError:
            pass

    # this is necessary in order for Blender API to see the GPUs
    bpy.context.preferences.addons['cycles'].preferences.get_devices()
    # for device in bpy.context.preferences.addons['cycles'].preferences.devices:
        # device.use = True
    scene.view_layers['ViewLayer'].use_pass_object_index = True
    # scene.view_layers['ViewLayer'].use_pass_material_index = True
    scene.view_layers['ViewLayer'].use_pass_mist = True

    bpy.data.scenes["Scene"].frame_end = kwargs['frames']

    ############################################################################
    #                              TEXTURES                                    #
    ############################################################################
    # remove all default materials residing in blender
    [bpy.data.materials.remove(mat) for mat in bpy.data.materials]
    # load textures from assets file
    add_materials(kwargs['textures'])
    # for im in bpy.data.images:
    #     try:
    #         ratio = im.size[0]/im.size[1]
    #     except:
    #         continue
    #     else:
    #         im.scale(1024, math.ceil(1024/ratio))
    # bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath('bridge_materials_mid_res.blend'))

    # When a material is an asset, a fake user is automatically assigned to it.
    # This leads to all the materials being saved in the blendfile, even without them
    # being used.[if the file is appended and not linked, the images still get saved,
    # even if the material has no users].
    for mat in bpy.data.materials:
        mat.use_fake_user = False
    ############################################################################
    #                          SCENE NODE GRAPH                                #
    ############################################################################
    # define compositing node graph
    init_compositing_graph(scene=scene,
                        categories=kwargs['class_dict'],
                        savepath=kwargs['savefolder'])

    ############################################################################
    #                             SKY                                         #
    ############################################################################
    # add sky to the scene
    add_sky(scene)


def object_vertices(obj, position=None, corner=False):
    """
    returns a matrix containing a subset of an object's vertices'
    global coordinates, that satisfy the position requirements
    (position is defined with respect to the object's origin location)
    If corner is set, only corner coordinates are returned
    """
    # all object vertices' global coordinates
    if obj.type != 'MESH':
        return None
    # # bpy version
    # coords = np.asarray([obj.matrix_world@v.co for v in obj.data.vertices])
    # bmesh version
    coords = []
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    if corner:
        coords = np.asarray([obj.matrix_world@v.co for v in bm.verts if len(v.link_edges)==3])
    else:
        coords = np.asarray([obj.matrix_world@v.co for v in bm.verts])
    bm.clear()

    if position is None:
        return coords
    if 'west' in position:
        coords = np.asarray([x for x in coords if x[0]>=obj.location[0]])
    if 'east' in position:
        coords = np.asarray([x for x in coords if x[0]<=obj.location[0]])
    if 'south' in position:
        coords = np.asarray([x for x in coords if x[1]>=obj.location[1]])
    if 'north' in position:
        coords = np.asarray([x for x in coords if x[1]<=obj.location[1]])
    if 'top' in position:
        coords = np.asarray([x for x in coords if x[2]>=obj.location[2]])
    if 'bottom' in position:
        coords = np.asarray([x for x in coords if x[2]<=obj.location[2]])

    return coords


def vertices_by_relative_position(coords, position):
    """
    get the subset of coordinates that corresponds to the position
    description, with respect to the center of gravity of the vertices.
    """
    coords = np.asarray(coords)
    cog = np.mean(coords, axis=0)
    if 'west' in position:
        coords = np.asarray([x for x in coords if x[0]>=cog[0]])
    elif 'east' in position:
        coords = np.asarray([x for x in coords if x[0]<=cog[0]])
    cog = np.mean(coords, axis=0)
    if 'south' in position:
        coords = np.asarray([x for x in coords if x[1]>=cog[1]])
    elif 'north' in position:
        coords = np.asarray([x for x in coords if x[1]<=cog[1]])
    cog = np.mean(coords, axis=0)
    if 'top' in position:
        coords = np.asarray([x for x in coords if x[2]>=cog[2]])
    elif 'bottom' in position:
        coords = np.asarray([x for x in coords if x[2]<=cog[2]])

    return coords


def ojb2hexahedronCoords(obj):
    """
    this function takes a mesh object with an arbitrary number of
    vertices and returns an array of 8 3d corrdinates, so that
    (for a y-forward, z-up view):
    0: back top right
    1: back top left
    2: back bottom left
    3: back bottom right
    4: front top right
    5: front top left
    6: front bottom left
    7: front bottom right
    """
    # all object vertices' global coordinates
    if obj.type != 'MESH':
        return None
    # # bpy version
    # coords = np.asarray([obj.matrix_world@v.co for v in obj.data.vertices])
    # bmesh version
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    # get mesh's corner coordinates
    coords = np.asarray([obj.matrix_world@v.co for v in bm.verts if len(v.link_edges)==3])
    bm.clear()
    return np.array(simulate_hexahedral_coords(coords))


@iterate(8)
def simulate_hexahedral_coords(hex_i, coords):
    """
    takes am array of 3d coords, and
    an index indicating one of the following positions:
    (for a y-forward, z-up view):
    0: back top right
    1: back top left
    2: back bottom left
    3: back bottom right
    4: front top right
    5: front top left
    6: front bottom left
    7: front bottom right
    and returns the coords of the position corresponding to hex_i
    INPUTS:
    coords: 8x3 array of 3d coords represending a hexahedron
    hex_i: integer in {0,...,7}
    OUTPUT:
    index: integer in {0, ... len(coords)} pointing to the
            simulated position
    NOTE: we chose to first deal with the axis that has the highest
        margin, to avoid mistakes in the relative position that can be
        caused by extreme cases of objects' orientations
    """
    limits = np.max(coords, axis=0) - np.min(coords, axis=0)
    cog = np.median(coords, axis=0) #center of gravity
    axis_order = np.argsort(limits) # indices that sort limits in ascending order
    for axis in axis_order[::-1]:
        if axis==0:
            if hex_i in [0,3,4,7]: # right-east
                coords = coords[coords[:,0]<=cog[0]]
            else: # left-west
                coords = coords[coords[:,0]>=cog[0]]
        elif axis==1:
            if hex_i in np.arange(4): # back-north
                coords = coords[coords[:,1]<=cog[1]]
            else: # front-south
                coords = coords[coords[:,1]>=cog[1]]
        elif axis==2:
            if hex_i in [2,3,6,7]: # top
                coords = coords[coords[:,2]<=cog[2]]
            else: # bottom
                coords = coords[coords[:,2]>=cog[2]]
        cog = np.median(coords, axis=0)
    # if more than one coordinates correspond, arbitrary choice
    # could be a max or min on some dimension at some point
    return coords[0, ...]


def check_vertex_visibility(objects=None, scene=None, camera=None, min_coverage=0):
    """
    check whether any object is visible from the camera
    INPUTS:
    @objects: list of objects whose vertices' visibility is checked. If None all objects in the scene are taken
    @scene:
    @camera: camera in whose view the vertex visibility is evaluated. If None the first camera object is taken
    @min_coverage: float in [0,1], denoting the minimum percentage of all the objects vertices that should be
                    visible
    OUTPUTS:
    1. boolean, assesing the vertex visibility
    2. integer, number of visible vertices
    3. integer, total number of (unique) vertices
    """
    objects = bpy.data.objects if objects is None else objects
    scene = bpy.context.scene if scene is None else scene
    camera = bpy.data.objects['Camera'] if camera is None else camera
    assert 0<=min_coverage<=1, 'min_coverage should be a value in [0,1], or None'

    # vertex_coords = [v.co for mesh in bpy.data.meshes for v in mesh.vertices]
    vertex_coords = [obj.matrix_world@v.co for obj in objects if obj.type=='MESH' for v in obj.data.vertices]
    # some vertices are doubles, since the same vertex location belongs to more than one objects
    vertex_coords = set([x.freeze() for x in vertex_coords])
    # check if there is any object vertex in the camera view
    coords_camera_view = [world_to_camera_view(scene, camera, coord) for coord in vertex_coords]
    # boolean array
    in_camera_view = [(0<coord[0]<1 and 0<coord[1]<1 and camera.data.clip_start<coord[2]<camera.data.clip_end)
                        for coord in coords_camera_view]

    # in_view = [obj.visible_camera() for obj in objects if obj.type=='MESH'] # does not work, not sure what exactly visible_camera returns
    if np.sum(in_camera_view)/len(vertex_coords)>=min_coverage:
        return True, np.sum(in_camera_view), len(vertex_coords)
    else:
        return False, np.sum(in_camera_view), len(vertex_coords)


def separate_by_material(object):
    """
    seperates the input object's mesh by material, and
    returnes the produced objects
    """
    # If object type is mesh and mode is set to object
    bpy.context.view_layer.objects.active = object
    # bpy.data.objects[obj_object.name].select_set(True)
    if object.type == 'MESH' and bpy.context.mode == 'OBJECT':
        # Edit Mode
        bpy.ops.object.mode_set(mode='EDIT')
        # merge edges by distance so that the doubles are removed
        bpy.ops.mesh.remove_doubles(threshold=0.1)
        # Seperate by material
        bpy.ops.mesh.separate(type='MATERIAL')
        # Object Mode
        bpy.ops.object.mode_set(mode='OBJECT')
    objs = bpy.context.selected_objects
    del object

    return objs


@forall
def subdivide_meshes(obj, n_cuts=1):
    # subdivide meshes so that the vertex visibility works better
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.subdivide_edges(bm,
                              edges=bm.edges,
                              cuts=n_cuts, # if 0 no cuts are made
                              use_grid_fill=True,
                              )
    # Write back to the mesh
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.clear()
    # bm.free()


@forall
def simplify_meshes(obj):
    # unsubdivide so that there is no increased load
    # actually this operation is a bit heavy for a large number of vertices
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.dissolve_limit(bm, angle_limit=np.deg2rad(1.7), verts=bm.verts, edges=bm.edges)
    # Write back to the mesh
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.clear()


@forall
def remove_doubles_from_meshes(obj, dist=0.3):
    """
    remove vertices that are closer than dist from the objects' mesh
    """
    if obj.type != 'MESH':
        return
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    # Write back to the mesh
    bm.to_mesh(obj.data)
    obj.data.update()
    bm.clear()


def component_description_from_material(objects):
    """
    generates a dictionary with category names as keys, and category ids as values
    the material of each one of the objects signifies its category.also renames the
    objects
    """
    cat_desc = []
    for obj in objects:
        match = re.search('material_(Label_([a-zA-Z]+)_Name_)?', obj.active_material.name)
        label = match.groups()[1]
        # clean_name = re.sub('material_(Label_[a-zA-Z]+_Name_)?', '', obj.active_material.name)
        clean_name = obj.active_material.name.replace(match.group(), '')
        # obj.name = unidecode.unidecode(obj.active_material.name.replace('material_', '')).lower() # change object name to reflect its category
        obj.name = unidecode.unidecode(clean_name) # change object name to reflect its category
        obj.data.name = obj.name
        label = obj.name if label is None else label
        cat_desc.append(label)
    cat_desc = list(set(cat_desc))
    cat_dict = {desc: i+1 for i,desc in enumerate(cat_desc)}
    del cat_desc
    return cat_dict


def add_camera():
    """
    adds a camera to the scene.
    The camera is set to be about 20 m in front of the bridge,
    looking at its entrance.
    The bridge length is considered to the the y axis
    """
    abutment_walls = next(obj for obj in bpy.data.objects
                                if 'abutment' in obj.name.lower())
    # front = max([v.co[1] for v in abutment_walls.data.vertices])
    # place camera in the middle of the road, 1.7m above the road, in front of the
    # bridge
    coords = [abutment_walls.matrix_world@v.co for v in abutment_walls.data.vertices]
    coords_east = vertices_by_relative_position(coords, 'bottom east')
    ind = np.argmin(coords_east[..., 1])
    point_br = coords_east[ind]
    ind = np.argmax(coords_east[..., 1])
    point_fr = coords_east[ind]
    # extrapolate
    fr_ext, _ = algebra_utils.extend_3Dline_along_y(point_br, point_fr, 20)
    location = [0, fr_ext[1], fr_ext[2]+1.7]
    rotation_euler = np.deg2rad([90, 0, 180])
    bpy.ops.object.camera_add(location=location, rotation=rotation_euler)
    bpy.context.scene.camera = bpy.context.object

    # camera object parameters
    camera = bpy.data.objects['Camera']
    # display a line for the mist start and end point in the blender file
    camera.data.show_mist = True
    # camera.lens = # lens value in mm (def 50mm)
    # camera.angle = # camera lens fov (def 0)
    # camera.ortho_scale = # Orthographic Camera scale (similar to zoom) float in [0, inf], default 6.0
    return camera


def add_sky(scene):
    """
    add sky to the scene using the Dynamic sky addon
    """
    # enable dynamic sky add on
    bpy.ops.preferences.addon_enable(module='lighting_dynamic_sky')
    # create dynamic sky
    bpy.ops.sky.dyn()
    scene.world = bpy.data.worlds['Dynamic_1']


def set_sky(scene):
    """
    randomly initialise sky parameters based on Dynamic sky addon
    """
    # set sky parameters
    # scene brightness, float default 1
    scene.world.node_tree.nodes['Scene_Brightness'].inputs['Strength'].default_value = \
    np.random.uniform(0.5, 5)
    # sky color (RGBA float 0-1) default [0.005, 0.435, 1, 1]
    scene.world.node_tree.nodes['Sky_and_Horizon_colors'].inputs[1].default_value = \
    [np.random.uniform(0.005, 0.3) , np.random.uniform(0.3, 1), np.random.uniform(0.8, 1), 1]
    # horizon color,  RGBA default [0.926, 0.822, 0.822, 1]
    # scene.world.node_tree.nodes['Sky_and_Horizon_colors'].inputs[2].default_value =
    # cloud color  RGBA float (0-1) default [1, 1, 1, 1]
    # scene.world.node_tree.nodes['Cloud_color'].inputs[1].default_value = \
    # cloud opacity float 0-1,  default 1
    scene.world.node_tree.nodes['Cloud_opacity'].inputs[0].default_value = \
    np.random.uniform(0, 1)
    # cloud density float 0-1,  default 0.267
    scene.world.node_tree.nodes['Cloud_density'].inputs[0].default_value = \
    np.random.uniform(0, 1)
    # sun color  RGBA float (0-1) default [0.5, 0.5, 0.5, 1]
    scene.world.node_tree.nodes['Sun_color'].inputs[1].default_value = \
    np.append(np.repeat(np.random.uniform(0,1), 3), 1)
    # sun intensity float 0-1, default 1
    scene.world.node_tree.nodes['Sun_value'].inputs[1].default_value = \
    np.random.uniform(0, 1)
    # sun direction, # vector xyz range [-1, 1]
    scene.world.node_tree.nodes['Sky_normal'].inputs['Normal'].default_value = \
    [np.random.uniform(0,1) for i in range(3)]
    # shadow hardness  float default 1 [0-1], 1: more sharp shadows
    scene.world.node_tree.nodes['Soft_hard'].inputs[0].default_value = \
    np.random.uniform(0,1)

    # MIST (mist pass property)
    scene.world.mist_settings.start = np.random.randint(3, 25)
    scene.world.mist_settings.depth = np.random.randint(30, 100)


@optional
@verify_format('.blend')
def add_materials(filepath=None, link=True):
    """
    finds all textures that are included in a blend file (suggestion: download from polyhaven)
    and appends them to the file as materials. Only textures that are marked as
    assets in the blend file are appended.
    @link: a boolean, if True the materials are linked, if false they are appended
    """
    # asset_libraries = bpy.context.preferences.filepaths.asset_libraries
    # ind = [i for i,x in enumerate(asset_libraries) if x.name =='bridge_materials']
    # if len(ind)>0:
    #     [bpy.ops.preferences.asset_library_remove(index=i) for i in ind]
    # bpy.ops.preferences.asset_library_add()
    # asset_libraries[-1].name = 'bridge_materials'
    # asset_libraries[-1].path = folderpath
    # # bpy.context.space_data.params.asset_library_ref = 'bridge_materials'
    # blend_files = glob.glob(os.path.join(folderpath, '*.blend'))
    # for blend_file in blend_files:
    with bpy.data.libraries.load(os.path.abspath(filepath),link=link, assets_only=True) as (data_from, data_to):
        data_to.materials = data_from.materials


@optional
def set_materials(materials, objs):
    """
    This function selects and applies an appropriate material to each object
    INPUTS:
    @objs: a list of blender objects
    @materials: a list of
    """
    material = random.choice(materials)
    for obj in objs:
        # adjust mapping scale for wide objects
        if any([x in obj.name for x in ['edge_beam', 'deck']]):
            obj.active_material = material.copy()
            coords = object_vertices(obj, corner=True)
            width = np.amax(coords[..., 0]) - np.amin(coords[..., 0])
            height = np.amax(coords[..., 2]) - np.amin(coords[..., 2])
            obj.active_material.node_tree.nodes['Mapping'].inputs['Scale'].default_value[0] = width/height
        else:
            obj.active_material = material
        if 'ground' in obj.name:
            ground_nodes = obj.active_material.node_tree.nodes
            # change texture node output from UV to generated
            # (not sure why but UV renders as a single color for the ground)
            obj.active_material.node_tree.links.new(
                    input=ground_nodes['Texture Coordinate'].outputs['Generated'],
                    output=ground_nodes['Mapping'].inputs['Vector'])
            # Scale ground texture
            ground_nodes['Mapping'].inputs['Scale'].default_value = [5, 10, 5]
            # node = ground_nodes.new('ShaderNodeValue')
            # node.name = 'Scale'
            # node.outputs['Value'].default_value = 20
            # ground[side].active_material.node_tree.links.new(input=node.outputs['Value'],
            #         output=ground_nodes['Mapping'].inputs['Scale'])
            # node.parent = ground_nodes['Mapping'].parent


def init_compositing_graph(scene, categories, savepath='.'):
    """
    defines compositing node graph so that the rgb rendered image, along with
    a binary mask for every category are saved
    """
    scene.use_nodes = True
    scene_nodes = scene.node_tree.nodes
    # ID mask nodes, to create binary outputs as many as the categories
    for category, category_id in categories.items():
        node = scene_nodes.new('CompositorNodeIDMask')
        node.name = category
        scene_nodes[category].index = category_id
        scene.node_tree.links.new(input=scene_nodes['Render Layers'].outputs['IndexOB'],
                                    output=node.inputs['ID value'])

    # add mist pass node
    node = scene_nodes.new('CompositorNodeMixRGB')
    node.blend_type = 'ADD'
    node.name = 'Mist addition'
    scene.node_tree.links.new(input=scene_nodes['Render Layers'].outputs['Mist'],
                                output=node.inputs['Fac'])
    scene.node_tree.links.new(input=scene_nodes['Render Layers'].outputs['Image'],
                                output=node.inputs[1])

    # file output node, with several inputs
    # for the rendered image
    output_node = scene_nodes.new('CompositorNodeOutputFile')
    output_node.name = 'Bridge render'
    output_node.base_path = os.path.abspath(savepath)
    output_node.format.file_format = 'PNG'
    output_node.format.color_mode = 'RGB'
    # output_node.file_slots['Image'].path = os.path.join('images', 'image_') # name for file prefix
    output_node.file_slots['Image'].path = os.path.join('images', '')
    # scene.node_tree.links.new(input=scene_nodes['Render Layers'].outputs['Image'],
    #                         output=output_node.inputs['Image'])
    scene.node_tree.links.new(input=scene_nodes['Mist addition'].outputs['Image'],
                            output=output_node.inputs['Image'])

    # save the image without the mist, for demonstration purposes
    output_node.file_slots.new(name='no mist')
    output_node.file_slots['no mist'].format.file_format = 'PNG'
    output_node.file_slots['no mist'].format.color_mode = 'RGB'
    output_node.file_slots['no mist'].path = os.path.join('images_without_mist', '') # name for file prefix
    scene.node_tree.links.new(input=scene_nodes['Render Layers'].outputs['Image'],
                            output=output_node.inputs['no mist'])

    # for each category
    for category in categories.keys():
        output_node.file_slots.new(name=category)
        output_node.file_slots[category].format.file_format = 'PNG'
        output_node.file_slots[category].format.color_mode = 'BW'
        output_node.file_slots[category].path = os.path.join('masks', '{}'.format(category), 'mask_') # subpath
        scene.node_tree.links.new(input=scene_nodes[category].outputs['Alpha'],
                            output=output_node.inputs[category])


def set_compositing_graph(scene=None, savepath='.', bridgename='somebridge'):
    """
    defines compositing node graph so that the rgb rendered image, along with
    a binary mask for every category are saved
    """
    scene = bpy.context.scene if scene is None else scene
    tree = scene.node_tree

    scene_nodes = scene.node_tree.nodes
    # scene_nodes['Bridge render'].base_path = savepath
    for slot in scene_nodes['Bridge render'].file_slots:
        if 'images' in slot.path:
            slot.path = os.path.join(os.path.dirname(slot.path), f'image_{bridgename}_')
        elif 'masks' in slot.path:
            slot.path = os.path.join(os.path.dirname(slot.path), f'mask_{bridgename}_')

    # set cloud color for mist
    sky_color = scene.world.node_tree.nodes['Sky_and_Horizon_colors'].inputs[1].default_value
    scene_nodes['Mist addition'].inputs[2].default_value = sky_color


def rename_objects_from_material(objects):
    """
    renames the object as the the material of each one of the objects signifies
    its category and name. Also creates a collection for each label, and puts
    inside it the objects that belong to this label.
    naming convention for material: material_Label_xxx_Name_yyy
    """
    cat_desc = []
    for obj in objects:
        match = re.search('material_(Label_([a-z_]+)_Name_)?',
                obj.active_material.name,
                flags=re.I)
        label = match.groups()[1]
        label = obj.name if label is None else label
        if label.lower() not in [x.name.lower() for x in bpy.data.collections]:
            coll = bpy.data.collections.new(label.lower())
            bpy.context.scene.collection.children.link(coll)
        # clean_name = re.sub('material_(Label_[a-zA-Z]+_Name_)?', '', obj.active_material.name)
        clean_name = obj.active_material.name.replace(match.group(), '')
        obj.name = unidecode.unidecode(clean_name) # change object name to reflect its category
        obj.data.name = obj.name
        # put object in the collection corresponding to its class
        try:
            [x.objects.unlink(obj) for x in obj.users_collection]
            bpy.data.collections[label].objects.link(obj)
        except:
            continue


def bridgeObject2componentObjects(obj, classes_dict=None):
    """
    this function takes a single object that represents a bridge,
    with different materials for each component, and breaks it up
    into several objects, one per component. It also adds a
    pass index to each object in order to generate the annotations.
    """
    # seperate mesh by material
    objs = separate_by_material(obj)
    del obj

    rename_objects_from_material(objs)
    if classes_dict is None:
        labels = [x.name for x in bpy.data.collections]
        labels = list(set(labels))
        classes_dict = {desc: i+1 for i,desc in enumerate(labels)}

    # remove the label materials from the file, it has served its purpose
    # to avoid duplicates (with names traverse, traverse.001) in the next bridge
    for obj in objs:
        bpy.data.materials.remove(obj.active_material)
    # remove materials that were created specifically for another bridge's proportions
    [bpy.data.materials.remove(mat) for mat in bpy.data.materials if mat.asset_data is None]
    # clean up empty collections
    [bpy.data.collections.remove(x) for x in bpy.data.collections if len(x.all_objects)==0]
    # assign pass index property to each object to signify its class
    for obj in objs:
        if obj.users_collection[0].name in classes_dict.keys():
            obj.pass_index = classes_dict[obj.users_collection[0].name]
        else:
            obj.pass_index = 0 # this is the default pass index of all objects    pass


def infer_class_dict():
    """
    This function infers and returns a semantic class dictionary from the categories
    and pass index of a blender file
    """
    class_dict = {obj.users_collection[0].name: obj.pass_index
                    for obj in bpy.data.objects
                    if obj.pass_index!=0}
    return class_dict


def generate_renders(objects=None, n_frames=1, min_coverage=0):
    """
    this function creates and saves a synthetic dataset
    (images and groundtruths) based on certain specifications
    INPUTS:
    @objects: list of objects whose vertices' visibility is checked. If None all objects in the scene are taken
    @scene:
    @n_frames: integer denoting how many frames should be (attempted to be) generated. If some of the frames
                do not comply with the min_coverage critetio, they will not be generated
    @min_coverage: float in [0,1], denoting the minimum percentage of all the objects vertices that should be
                    visible
    OUTPUTS:

    """
    # enable smooth shading and auto smooth for better render experience
    for mesh in bpy.data.meshes:
        mesh.use_auto_smooth=True

    objects = [x for x in bpy.data.objects if x.type=='MESH'] if objects is None else objects
    scene = bpy.context.scene

    camera = bpy.data.objects['Camera']
    # angle of view parameters for camera (meters and radians)
    camera_init_location = copy.deepcopy(camera.location)
    camera_init_rotation = copy.deepcopy(camera.rotation_euler)

    # make sure sampled camera locations simulate an inspector that is on the road
    road = bpy.data.objects['road']
    decks = [obj for obj in bpy.data.objects if 'deck' in obj.name]
    road_coords = np.asarray([v.co for v in road.data.vertices])
    deck_coords = np.asarray([deck.matrix_world@v.co for deck in decks
                                                    for v in deck.data.vertices])

    x_margin = [min(road_coords[:,0]), max(road_coords[:,0])]
    y_margin = [min(deck_coords[:,1]-10), max(deck_coords[:,1])+10]
    z_margin = [camera_init_location[2]-0.3, camera_init_location[2]+0.3]

    camera_locations, camera_rotations = algebra_utils.sample_positions(
                                init_location=camera_init_location,
                                init_rotation=np.rad2deg(camera_init_rotation),
                                location_ranges=[x_margin, y_margin, z_margin], #[x_margin, y_margin, 0.3]
                                rotation_ranges=[80, 5, 80], #10, 5, 45
                                n_samples=n_frames)


    render_cnt = 0
    for frame in range(n_frames):
        # step forward in time
        scene.frame_set(frame) # necessary in order to save all frames
        # position camera
        camera.location = camera_locations[frame] if n_frames!=1 else camera_init_location
        camera.rotation_euler = np.deg2rad(camera_rotations[frame]) if n_frames!=1 else camera_init_rotation
        bpy.context.view_layer.update() # necessary for world_to_camera_view to work

        check, n_visible, n_vertices = check_vertex_visibility(objects, scene, camera, min_coverage=min_coverage)

        if check:
            start_render_only = time.time()
            try:
                bpy.ops.render.render()
            except:
                continue
            render_cnt += 1

    # return camera to initial position
    # camera.location = camera_init_location
    # camera.rotation_euler = camera_init_rotation
    return render_cnt
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# ----------------------------DEBUGGING-----------------------------
# for attr in dir(variable):
#     print('{} : {}'.format(attr, getattr(variable, attr)))
# ------------------------------------------------------------------
