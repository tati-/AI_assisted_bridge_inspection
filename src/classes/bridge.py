import os
import re
import sys
import bpy
import json
import math
import random
import warnings
import numpy as np
import numpy.typing as npt
from pathlib import Path
from typing import Literal

import modules.utils as utils
import modules.blender_utils as bl
import modules.algebra_utils as alg
import modules.ifc_utils as ifc_utils
import modules.optimization_utils as optim
import modules.translate_utils as translate
from classes.building_block import Hexahedron, Constraint


class BridgeModel:
    """
    This class represents a bridge as a collection of hexahedra.
    The hexahedra dimensions and dependencies are given via a json file
    """
    def __init__(self,
                json_data_path: str,
                json_param_path: str=None):
        """
        @json_data_path: a json file containing all the parameters needed to create
                a bridge .obj model, in the form of a list of building blocks(cubes)
        @json_param_path: a json file containing the parameter margins, in order to
                        to randomize the bridge structure
        !!! The wing walls initial placement is 90 degrees with respect to the
            abutment (piedroit) and facing towards the west.
        NOTE: constraints in .json file are pointers to building blocks (zero-based)
        Constraint conventions:
        ID in Json file
        -1: no selection
        0: 1st list element
        2: 2nd list element
        ...
        To simplify the input, the first field is empty, in order to be able to
        remove a potential selection
        SO
        Value -1 or null = no constraints
        Value 0 = parent element 0
        Value 1 = parent element 1
        etc
        """
        self.building_blocks = list()
        with open(json_data_path) as json_file:
            self.data = json.load(json_file)
        """
        test = utils.json_lowercase(json_data_path)
        # test = translate.translate_json(json_data_path)
        with open(Path(json_data_path).with_stem(f'{Path(json_data_path).stem}_en'), 'w') as fp:
            json.dump(test, fp, indent=2)
        """
        # the json file is in french, here I translate it to have some more
        # representative keys in english
        # self.data = translate.translate_json(json_data_path)
        self.initialize_visibility()
        if json_param_path is not None:
            with open(json_param_path) as json_file:
                rules_dict = json.load(json_file)
            # rules_dict = translate.translate_json(json_param_path)
            self.randomize_data(rules_dict)
        self.constraints_from_dict()
        self.blocks_from_dict()
        self.initialize_mesh_data()


    def initialize_visibility(self, value: bool=True):
        """
        sets all building blocks to the same value
        """
        for item in self.data:
            if 'visible' not in item.keys():
                item['visible'] = value


    def initialize_mesh_data(self):
        # self.mesh ='mtllib ./Label.mtl\n'
        self.normals=''
        self.texture ='vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n'
        self.faces=''
        self.materials ='newmtl material_A\nd 0.5\n'


    def dependent_var_no_bounds(self, rule: dict) -> float:
        """
        this function calculates the random value to be assigned to a dimension
        or angle property, based on the rule defined in the  entry of
        the json file defining the parameters dependency.
        This function does not take into account the min and max bounds.
        INPUTS:
        @rule : dictionary object retrieved from the json file, denoting the
                independent variable and the first degree coefficient
        OUTPUTS:
        @new_value: the value of the property if it is dependent on another
                    property, None otherwise
        """
        if rule['coefficient'] is None:
            return None
        assert len(rule['coefficient'])==len(rule['equation']), 'the number of coefficients should be the same as that of independent variables'
        new_value = 0
        for i, var in enumerate(rule['equation']):
            mother_block = [b for b in self.data if b['label']==var['label']]
            if 'name' in var.keys():
                mother_block = [b for b in mother_block if b['name']==var['name']]

            if len(mother_block)==0: continue

            independent_value = mother_block[0]
            for d in var['constraint']:
                # list denoting the path of nested dictionaries keys that retrieve
                # the independent variable value
                independent_value = independent_value[d]

            # if the coeffient is fixed, a single value is given.
            # if it is not fixed, an array of the min and the max value of the
            # coefficient is given
            if rule['coefficient'][i]['fixed']:
                coef = rule['coefficient'][i]['value']
            else:
                coef = random.uniform(
                            rule['coefficient'][i]['value'][0],
                            rule['coefficient'][i]['value'][1])
            new_value += coef*independent_value
        if 'constant' in rule.keys():
            new_value += rule['constant']

        return new_value


    def min_max(self, rule: dict) -> tuple[float, float]:
        """
        this function calculates the min and max values of a property.
        The min and max can be either absolute values, or dependent on
        another property (relative min and max)
        INPUTS:
        @rule : dictionary object retrieved from the json file, denoting the
                absolute and relative min and max
        OUTPUTS:
        @ min_, max_: the min and max values taking all restrictions into account.
        If there is no restriction, -inf or inf are returned.
        """
        min_, max_ = -math.inf, math.inf
        if 'min' in rule.keys():
            min_ = rule['min']
        if 'max' in rule.keys():
            max_ = rule['max']
        if 'rmin' in rule.keys():
            rmin = self.dependent_var_no_bounds(rule['rmin'])
            if rmin is not None:
                min_ = max(min_, rmin)
        if 'rmax' in rule.keys():
            rmax = self.dependent_var_no_bounds(rule['rmax'])
            if rmax is not None:
                max_ = min(max_, rmax)

        return min_, max_


    def randomize_data(self, constraints: npt.ArrayLike):
        """
        this function uses the parameter margins to randomize the bridge
        dimensions
        @constraints: list of dictionary items
        """
        for constraint in constraints:
            affected_data = [i for i, x in enumerate(self.data) if x['label']==constraint['label']]
            if 'name' in constraint.keys():
                affected_data = [i for i in affected_data if self.data[i]['name']==constraint['name']]
            if len(affected_data)==0: continue
            # randomly select which blocks will be visible
            for index in affected_data:
                if not constraint['necessary'] and np.random.choice([True, False], p=[0.2, 0.8]):
                    self.data[index]['visible'] = False
                    continue
            # calculate a random value for each element based on the rules
            for property, property_content in constraint.items(): # property should be dimensions or angles
                if property in ['label', 'name', 'necessary']:
                    continue
                for specific_property, rules in property_content.items(): # specific property should be width, length, height for dimensions, heading, roll or tilt for angle
                    for rule in rules:
                        min_, max_ = self.min_max(rule)
                        # here we could potentially add a pdf implementation
                        if rule['coefficient'] is None:
                            if 'maxprobability' in rule.keys():
                                new_value = np.random.choice([max_, random.uniform(min_, max_)],
                                             p=[rule['maxprobability'], 1-rule['maxprobability']])
                            else:
                                new_value = random.uniform(min_, max_)
                        else:
                            new_value = self.dependent_var_no_bounds(rule)
                            new_value = np.clip(new_value, min_, max_)
                        if new_value is None: continue
                        if new_value==-math.inf or new_value==math.inf:
                            warnings.warn('\n##-----## \nWarning: {} of {} is set' \
                                        'to infinity for element {}. This value will be '\
                                        'ignored\n##-----##'.format(
                                        property, specific_property, self.data[index]['name']))
                            continue
                        for index in affected_data:
                            for more_specific_property in rule['dependent_var']: # more specific property should be d1, d2, offset or v
                                self.data[index][property][specific_property][more_specific_property] = new_value



    def get_dimension_parent(self,
                            index: int,
                            dimension: str) -> int:
        """
        returns the index of the building block whose given dimension
        should be used
        """
        i = index
        # find dependency for dimension measurements
        while self.data[i]['dimensions'][dimension]['constraint']>=0:
            if self.data[i]['dimensions'][dimension]['constraint']==i:
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['dimensions'][dimension]['constraint']
        return i


    def get_offset_parent(self,
                        index: int,
                        dimension: str):
        """
        returns the index of the building block whose offset for the given dimension
        should be used
        """
        i = index
        while self.data[i]['dimensions'][dimension]['offset_constraint']>=0:
            if self.data[i]['dimensions'][dimension]['offset_constraint']==i:
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['dimensions'][dimension]['offset_constraint']
        return i


    def get_rotation_parent(self,
                            index: int,
                            rot_axis: str):
        """
        returns the index of the building block whose rotation angle for the given
        axis should be used
        """
        i = index
        while self.data[i]['angles'][rot_axis]['constraint']>=0:
            if self.data[i]['angles'][rot_axis]['constraint']==i:
                # normally we should never go in here
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['angles'][rot_axis]['constraint']
        return i


    def get_dimensions(self, index: int) -> tuple:
        """
        This function returns the dimensions of the cube and the offset,
        taking into account the constraints
        INPUTS:
        @index: index of the building block whose rotation angle we want to
                find
        dimension: string, indication the dimension, in {'width', 'length', 'height'}
        OTPUTS:
        dim1: first dimension (towards positive face)
        dim2: second dimension (towards negative face)
        offset: offset, allowing the cube to tilt
        """
        dimensions = dict()
        for dimension in self.data[index]['dimensions'].keys():
            i_parent = getattr(self.constraints[index], dimension)
            dim1 = self.data[i_parent]['dimensions'][dimension]['d1']
            dim2 = self.data[i_parent]['dimensions'][dimension]['d2']

            # find dependency for offset (decalage)
            i_parent_offset = getattr(self.constraints[index], f'offset_{dimension}')
            offset = self.data[i_parent_offset]['dimensions'][dimension]['offset']

            dimensions[dimension] = (dim1, dim2, offset)

        # width, length, height
        return dimensions['width'], dimensions['length'], dimensions['height']


    def get_rotation_angles(self, index: int) -> tuple[float, float, float]:
        """
        This function returns the rotation angle in degrees (counterclockwise), taking into
        account the constraints
        rot_axis: string, indication of the rotation axis
        If the obj is loaded with Y forward, Z up, this corresponds to:
                z: 'heading' (bridge height - hauteur)
                y: 'tilt' (bridge length - largeur)
                x: 'roll' (bridge width - longueur)
        INPUTS:
        @index: index of the building block
        OUTPUTS:
        @ anglet, tilt, roll : corresponding angles in degrees
        """
        angles = dict()
        # i = index
        for rot_axis in self.data[index]['angles'].keys():
            i_parent = getattr(self.constraints[index], rot_axis)
            angles[rot_axis] = self.data[i_parent]['angles'][rot_axis]['v']

        # return x, y, z axis rotation angles in degrees
        return angles['roll'], angles['tilt'], angles['heading']


    def set_translation_offset(self, index: int) -> npt.ArrayLike:
        """
        This function calculates the new center of a child building block
        (assuming the old center is (0,0,0)), so that a child's node is connected
        to a parent's node, and adjusts its offset.
        The information on the parent building block, and the connecting nodes
        are given in the JSON file.
        INPUTS:
        @index: child building block's index
        OUTPUTS:
        @center_point_coords: numpy array containing the x,y,z coordinates where
            the child building block should be centered
        """
        self.building_blocks[index].offset = np.zeros(3)
        i = index
        # if this building block has no constraints, offset is (0,0,0)
        if (self.constraints[index].parent<0 or
            self.constraints[index].parent_node<0 or
            self.constraints[index].child_node<0):
            return

        while self.constraints[i].parent>=0:
            # index of parent building block
            i_parent=self.constraints[i].parent
            # assert self.data[index]['constraint']['parent_node']>=0, "Undefined parent node in the constraint declaration"
            # assert self.data[index]['constraint']['child_node']>=0, "Undefined child node in the constraint declaration"

            if self.constraints[i].parent_node>=0:
                # node (vertex) of parent building_block
                parent_node =self.constraints[i].parent_node
                parent_node_coords = self.building_blocks[i_parent].global_coords[parent_node]

            if self.constraints[i].child_node>=0:
                # node (vertex) of child building block
                child_node = self.constraints[i].child_node
                child_node_coords = self.building_blocks[i].global_coords[child_node]
                # child_node_coords = self.building_blocks[index].local_coords[child_node]

            # define the new center of the current building block, so that
            # the node1 (N1) of the parent block and the node2 (N2) of the
            # child block coincide
            self.building_blocks[index].offset += parent_node_coords - child_node_coords

            i = i_parent


    def constraints_from_dict(self):
        """
        This function creates a list of constraint objects describing the bridge
        building blocks constraints
        """
        self.constraints = list()
        for index, block in enumerate(self.data):
            constraint = Constraint(width=self.get_dimension_parent(index, 'width'),
                                    length=self.get_dimension_parent(index, 'length'),
                                    height=self.get_dimension_parent(index, 'height'),
                                    offset_width=self.get_offset_parent(index, 'width'),
                                    offset_length=self.get_offset_parent(index, 'length'),
                                    offset_height=self.get_offset_parent(index, 'height'),
                                    roll=self.get_rotation_parent(index, 'roll'),
                                    tilt=self.get_rotation_parent(index, 'tilt'),
                                    heading=self.get_rotation_parent(index, 'heading'),
                                    parent=block['constraint']['parent'],
                                    parent_node=block['constraint']['parent_node'],
                                    child_node=block['constraint']['child_node'])
            self.constraints.append(constraint)


    def blocks_from_dict(self):
        """
        This function creates a list of hexahedron objects describing the bridge
        building blocks
        """
        for index, building_block in enumerate(self.data):
            width, length, height = self.get_dimensions(index)
            rotx, roty, rotz = self.get_rotation_angles(index)
            self.building_blocks.append(Hexahedron(index, building_block['label'],
                                        building_block['name'], width, length, height,
                                        rotx, roty, rotz))
            # collapse all coordinates to 0 if block is not visible - this block
            # will not be created in the geometric file, it is kept for now to maintain the
            # self.data and self.building_blocks index correspondance
            if not self.data[index]['visible'] or not self.building_blocks[index].is_complete():
                continue

        for block in self.building_blocks:
            self.set_translation_offset(block.id)


    def update_child_blocks(self, parent_id:int):
        """
        Update a bridge's values taking into account the contstraints imposed
        by the parent_id block (update all blocks that are children of parent_id block)
        """
        if not self.building_blocks[parent_id].is_complete():
            pass

        # update children that have a dimension dependence
        for dimension in self.building_blocks[parent_id].dimensions.keys():
            child_blocks = [x for i,x in enumerate(self.building_blocks)
                            if getattr(self.constraints[i], dimension)==parent_id]
            for block in child_blocks:
                block.dimensions[dimension][:2] = self.building_blocks[parent_id].dimensions[dimension][:2]
            child_blocks = [x for i,x in enumerate(self.building_blocks)
                            if getattr(self.constraints[i], f'offset_{dimension}')==parent_id]
            for block in child_blocks:
                block.dimensions[dimension][2] = self.building_blocks[parent_id].dimensions[dimension][2]

        # update children that have a rotation dependence
        for rot_axis in self.building_blocks[parent_id].rotations.keys():
            child_blocks = [x for i,x in enumerate(self.building_blocks)
                            if getattr(self.constraints[i], rot_axis)==parent_id]
            for block in child_blocks:
                block.rotations[rot_axis] = self.building_blocks[parent_id].rotations[rot_axis]

        # update children that have a positional dependence
        child_blocks = [x for i,x in enumerate(self.building_blocks)
                        if self.constraints[i].parent==parent_id]
        for block in child_blocks:
            self.set_translation_offset(block.id)


    def update_data(self):
        """
        update self.data list of dictionaries to match the dimensions of the building blocks
        object
        # IMPORTANT: implies that there is a 1-1 correspondence between the self.data
        list and the self.building_blocks list
        """
        assert len(self.building_blocks)==len(self.data), f"Unable to match {len(self.building_blocks)} building blocks to {len(self.data)} dictionary objects"

        for i, block in enumerate(self.building_blocks):
            self.data[i]['name'] = block.name
            for dim in block.dimensions.keys():
                self.data[i]['dimensions'][dim]['d1'] = utils.numpy2native(block.dimensions[dim][0])
                self.data[i]['dimensions'][dim]['d2'] = utils.numpy2native(block.dimensions[dim][1])
                self.data[i]['dimensions'][dim]['offset'] = utils.numpy2native(block.dimensions[dim][2])
            for rot in block.rotations.keys():
                self.data[i]['angles'][rot]['v'] = utils.numpy2native(block.rotations[rot])
            self.data[i]['visible'] = block.visible


    def characterize_wing_walls(self):
        """
        infer if wing walls are "mur en aile" or "mur en retour",
        and add an indication in their name
        mur en retour: 90 degree angle with the abutment(piedroit)
        mur en aile: all other angles
        """
        wing_walls = [blo for blo in self.building_blocks if blo.label=='wing_wall']
        for wall in wing_walls:
            # define the abutment that the wall is attached on
            # the abutments last letter (e or w) should correspond the
            # one of the wall
            abutment = [blo for blo in self.building_blocks if blo.label=='abutment'
                        and blo.name[-1] == wall.name[-1]][0]
            if np.any([math.isclose(wall.rotations['heading']-abutment.rotations['heading'], i, abs_tol=5)
                        for i in [0, 180, -180, 360]]):
                cl = 'en_retour'
            else:
                cl = 'en_aile'
            # add the characterization to the wall's name
            wall.name = re.sub(f'(wing_wall)(-[nsew]{{1,2}})$', rf'\1_{cl}\2', wall.name)

            self.update_data()


    @property
    def mesh(self):
        """
        This function processes the JSON file and creates the obj mesh data
        """
        # initialize mesh data to clean any previous data
        mesh = 'mtllib ./Label.mtl\n'
        self.initialize_mesh_data()

        labels = list()
        # face nodes: 0 indexed, used (node_index) to define the constraints
        # in the json file (zero-based)
        # face vertices, (one-based)
        # the indices correspond to the vertices defined in the obj mesh file
        # the correspondence with the local node indices is
        # node position - local node coord id - vertex index (for .obj file)
        # top back right      - 0 - 6
        # top back left       - 1 - 5
        # bottom back left    - 2 - 8
        # bottom back right   - 3 - 7
        # top front right     - 4 - 3
        # top front left      - 5 - 4
        # bottom front left   - 6 - 1
        # bottom front right  - 7 - 2
        face_desc = ['front', 'back', 'top', 'bottom', 'right', 'left']
        face_nodes = {
                     'front': np.asarray([6, 7, 4, 5]),
                     'back': np.asarray([1, 0, 3, 2]),
                     'top': np.asarray([5, 4, 0, 1]),
                     'bottom': np.asarray([2, 3, 7, 6]),
                     'right': np.asarray([4, 7, 3, 0]),
                     'left': np.asarray([6, 5, 1, 2])
                     }
        face_vertices = {
                         'front': np.asarray([1, 2, 3, 4]),
                         'back': np.asarray([7, 8, 5, 6]),
                         'top': np.asarray([4, 3, 6, 5]),
                         'bottom': np.asarray([8, 7, 2, 1]),
                         'right': np.asarray([2, 7, 6, 3]),
                         'left': np.asarray([8, 1, 4, 5])
                        }

        vis_index = -1 # index of visible element

        for index, block in enumerate(self.building_blocks):
            if block.visible:
                vis_index+=1
            else:
                continue
            # add material label if not already there
            label = f'Label_{block.label}_Name_{block.name}'
            if (label not in labels):
                labels.append(label)
                # self.materials += 'newmtl material_{}\nmap_kd Label/{}.png\n'.format(building_block['label'], building_block['label'])
                self.materials += f'newmtl material_{label}\n'
                self.faces += f'\n\n\nusemtl material_{label}\n'

            # create building block faces and add to mesh
            for i, face in enumerate(face_desc):
                face_coords = block.global_coords[face_nodes[face]]
                normal = alg.get_normal(face_coords)

                if face in ['front', 'back']:
                    mesh += 'v {} {} {}\nv {} {} {}\nv {} {} {}\nv {} {} {}\n'.format(*face_coords.flatten())
                self.normals += 'vn {} {} {}\n'.format(*normal)
                self.faces += 'f {}/1/{} {}/2/{} {}/3/{} {}/4/{}\n'.format(vis_index*8+face_vertices[face][0], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][1], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][2], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][3], vis_index*6+i+1)
                # self.faces += 'f {}/1 {}/2 {}/3 {}/4\n'.format(*(index*8+face_vertices[face]))

        mesh += self.normals + self.texture + self.faces

        return mesh


    def move(self, offset: npt.ArrayLike):
        """
        translate the bridge by offset
        INPUTS:
        @offset: 1x3 array, translation coordinates in x,y,z
        """
        for block in self.building_blocks:
            block.offset += offset


    def zero_mean(self):
        """
        recenter bridge so that it is around (0,0,0)
        """
        points = np.concatenate([block.global_coords for block in self.building_blocks])
        off = -np.mean(points, axis=0)
        self.move(off)


    def align_to_point_cloud(self, deck_points: npt.ArrayLike=None):
        """
        changes the offset of the bridge (of all the blocks) so that the
        deck of the bridge overlaps as much as possible with the point point cloud.
        INPUTS:
        @deck_points: nPoints x 3 array with coordinates of 3d points that belong
                to the deck
        """
        deck = [block for block in self.building_blocks if block.label=='traverse'][0]
        deck_points = deck.points if deck_points is None else deck_points
        res = optim.align_model_to_points(deck_points, deck.global_coords)
        self.move(res[0])
        print(f'Deck aligned with {(1-res[1])*100:.2f}% overlap.')


    @property
    def fv(self) -> npt.ArrayLike:
        """
        Convert bridge dimensions to a feature vector to be used in
        optimization algorithms
        """
        fv = np.zeros(12*len(self.building_blocks))
        for i, block in enumerate(self.building_blocks):
            fv[i*12: i*12+12] = block.fv

        return fv


    def from_fv(self, fv: npt.ArrayLike):
        """
        Convert feature vector to bridge dimensions
        INPUTS:
        @fv: array of length 12*len(building blocks)
        """
        assert len(fv)==12*len(self.building_blocks), f'Feature vector should be'\
                                                      f'of length {12*len(self.building_blocks)}, '\
                                                      f'{len(fv)} was given.'
        for i, block in enumerate(self.building_blocks):
            block.from_fv(fv[i*12: i*12+12])

        for block in self.building_blocks:
            self.update_child_blocks(parent_id=block.id)

        # update before exiting, otherwise the bridge data might not reflect the
        # building blocks' state
        self.update_data()


    @property
    def params(self):
        """
        returns a not nested dictionary with the parameters that the
        sizing process is supposed to define, and their current values
        No constraints are taken into account here
        """
        self.update_data()
        params = dict()
        for block in self.data:
            for desc, dim in block['dimensions'].items():
                params.update({'.'.join([block['name'], desc, i]): dim[i] for i in ['d1', 'd2', 'offset']})
            for desc, angle in block['angles'].items():
                params.update({'.'.join([block['name'], desc, 'v']): angle['v']})

        return params


    def from_params(self, params: dict):
        """
        convert a params dictionary to bridge dimensions.
        Such dictionary has keys of the form:
        [block_name].[width/lenght/height / roll/tilt/heading].[d1/d2/d3 / v]
        """
        fv = list()
        for block in self.building_blocks:
            try:
                fv.extend([params[f'{block.name}.{dim}.{el}']
                            for dim in ['width', 'length', 'height']
                            for el in ['d1', 'd2', 'offset']])
                fv.extend([params[f'{block.name}.{rot}.v']
                            for rot in ['roll', 'tilt', 'heading']])
            except:
                warnings.warn(f'One or more of the elements of {block.name} were'\
                                f' not found in the parameter dictionary. The result'\
                                f' of this function can therefore not be trusted.')
                continue
        self.from_fv(fv)


    def to_obj(self, objPath: str, labelPath: str=None):
        """
        saves mesh with material details as an .obj file
        INPUTS:
        @obj_path: path to OBJ file to be created
        @label_path: path to .mtl to be created, where the material information
                        will be stored
        """
        labelPath = Path(objPath).with_suffix('.mtl') if labelPath is None else labelPath
        with open(labelPath, "w+") as f:
            f.write(self.materials)
        with open(objPath, "w+") as f:
            f.write(self.mesh)


    def to_json(self, jsonPath: str):
        """
        saves the bridge data in json format
        """
        self.update_data()
        # create and save json file
        os.makedirs(os.path.dirname(jsonPath), exist_ok=True)
        with open(jsonPath, 'w') as fp:
            json.dump(self.data, fp, indent=2)


    def to_blender(self, blenderPath: str, keep_obj: bool=False):
        """
        saves the bridge as a blender file (need to pass via an obj file for
        that)
        """
        # create and save necessary obj file
        objPath = Path(blenderPath).with_suffix('.obj')
        labelPath = Path(blenderPath).with_suffix('.mtl')
        self.to_obj(objPath, labelPath)
        bl.clean_scene()

        bpy.ops.import_scene.obj(filepath=str(objPath), axis_forward='Y', axis_up='Z')
        obj = bpy.context.selected_objects[0] # returns an array of objects in the scene

        ############################################################################
        #                SPLIT TO SEMANTIC OBJECTS                                 #
        ############################################################################
        bl.bridgeObject2componentObjects(obj)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(blenderPath))

        if not keep_obj:
            os.remove(objPath)
            os.remove(labelPath)


    def to_ifc(self, ifcPath: str, keep_blender: bool=False):
        """
        saves the bridge as an IFC file, following the conventions defined in
        modules/ifc_utils.py (need to pass via a blender file for that)
        """
        blenderPath = Path(ifcPath).with_suffix('.blend')
        self.to_blender(blenderPath)

        # install blenderBIM addon if it does not exist
        ifc_utils.enable_blenderBIM()

        # open blender file
        bpy.ops.wm.open_mainfile(filepath=os.path.abspath(blenderPath))
        scene = bpy.context.scene

        # add fake ground to allow IFC placement
        abutments = [obj for obj in bpy.data.objects if 'abutment' in obj.name]
        bl.add_ground(abutments)

        # create bim project
        bpy.ops.bim.create_project()

        # assign ifc classes to objects
        ifc_utils.assign_ifc_classes(*bpy.data.objects)

        # save ifc file
        bpy.ops.export_ifc.bim(filepath=os.path.abspath(ifcPath))

        # remove bounding boxes that are automatically saved
        # for a reason that is not very clear to me yet, this import should be here,
        # in local scope. If put above with the imports, it breaks the blenderbim
        # import, I suppose blenderbim is using its own version of ifcopenshell
        import ifcopenshell
        ifc = ifcopenshell.open(ifcPath)
        for bbox in [rep for rep in ifc.by_type('IfcShapeRepresentation') if rep.RepresentationType == 'BoundingBox']:
            ifc.remove(bbox)
        ifc.write(ifcPath)

        if not keep_blender:
            os.remove(blenderPath)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# face_vertices_triang = {'front': np.asarray([[1, 2, 3], [4, 1, 3]]),
#                         'back': np.asarray([[5, 6, 7], [8, 5, 7]]),
#                         'top': np.asarray([[4, 3, 6], [5, 4, 6]]),
#                         'bottom': np.asarray([[8, 7, 2], [1, 8, 2]]),
#                         'right': np.asarray([[3, 2, 7], [6, 3, 7]]),
#                         'left': np.asarray([[1, 4, 5], [8, 1, 5]])
#                         }

# self.faces += 'f {}/1/{} {}/2/{} {}/3/{}\n'.format(index*8+face_vertices_triang[face][0,0], index*6+i+1,
#                                                 index*8+face_vertices_triang[face][0,1], index*6+i+1,
#                                                 index*8+face_vertices_triang[face][0,2], index*6+i+1)
# self.faces += 'f {}/1/{} {}/2/{} {}/3/{}\n'.format(index*8+face_vertices_triang[face][1,0], index*6+i+1,
#                                                 index*8+face_vertices_triang[face][1,1], index*6+i+1,
#                                                 index*8+face_vertices_triang[face][1,2], index*6+i+1)
