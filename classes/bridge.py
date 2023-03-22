import os
import sys
import pdb
import json
import math
import random
import argparse
import warnings
import numpy as np
sys.path.append('../modules')
import algebra_utils as alg

class Hexahedron:
    """
    this class represents a geometric element, an hexahedron used as a building
    block for the bridge
    """
    def __init__(self, id, name, width, length, height, rotx, roty, rotz):
        """
        @json_data: the list of dictionary items read from the diades json file
                    (check readme code for its structure)
        """
        # this id is the index of the current block in the blocks array
        # can I disassociate this?
        self.id = id;
        self.name = name
        # width, length, height
        self.dimensions = dict.fromkeys(['Longueur', 'Largeur', 'Hauteur'])
        # x, y, z axis rotation angles in degrees
        self.rotations = dict.fromkeys(['Roll', 'Tilt', 'Heading'])
        self.set_dimensions(width, length, height)
        self.set_rotations(rotx, roty, rotz)
        self.set_local_coords()
        offset = np.zeros(3)
        self.global_coords = self.local_coords + offset


    def set_dimensions(self, width, length, height):
        """
        This function sets the dimensions of the cube and the tilt(decalage),
        taking into account the constraints
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
        tilt: offset, allowing the cube to tilt
        """
        self.dimensions['Longueur'] = np.asarray(width)
        self.dimensions['Largeur'] = np.asarray(length)
        self.dimensions['Hauteur'] = np.asarray(height)


    def set_rotations(self, rotx, roty, rotz):
        """
        This function returns the rotation angle in degrees, taking into
        account the constraints
        rot_axis: string, indication of the rotation axis
        If the obj is loaded with Y forward, Z up, this corresponds to:
                z: 'Heading' (bridge height - hauteur)
                y: 'Tilt' (bridge length - largeur)
                x: 'Roll' (bridge width - longueur)
        INPUTS:
        @blocks: array of json information about all the building blocks
        """
        self.rotations['Roll'] = np.asarray(rotx)
        self.rotations['Tilt'] = np.asarray(roty)
        self.rotations['Heading'] = np.asarray(rotz)


    def get_local_node_coords(self, node_id):
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
                               third element: width offset
        @length: length-3 array, first element: left cube length
                               second element: right cube length
                               third element: length offset
        @height: length-3 array, first element: left cube height
                               second element: right cube height
                               third element: height offset
        """
        # the d1 and d2 (first two element) of the dimensions refer to the overal
        # hexahedron size, so we divide them by two, to get the coordinates
        width = self.dimensions['Longueur']/np.array([2,2,1])
        length = self.dimensions['Largeur']/np.array([2,2,1])
        height = self.dimensions['Hauteur']/np.array([2,2,1])
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


    def set_local_coords(self):
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
        INPUTS:
        @offset: numpy array of length 1x3, translation of hexahedron center in x, y, z axes
        OUTPUTS:
        @global_node_coords: 8x3 array, global hexahedron coordinates after rotation and offset
        """
        rot_angles = [self.rotations['Roll'], # x axis
                      self.rotations['Tilt'], # y axis
                      self.rotations['Heading'] # z axis
                     ]

        # get local node coordinates before rotation
        self.local_coords = np.asarray([self.get_local_node_coords(node) for node in range(8)])

        self.local_coords = np.apply_along_axis(alg.rotate_point,
                                                axis=1,
                                                arr=self.local_coords,
                                                rot_angles=rot_angles)


class BridgeModel:
    """
    This class represents a bridge as a collection of hexahedra.
    The hexahedra dimensions and dependencies are given via a json file
    """
    def __init__(self, json_data_path, obj_path, label_path, json_param_path=None):
        """
        @json_data_path: a json file containing all the parameters needed to create
                a bridge .obj model, in the form of a list of building blocks(cubes)
        @obj_path: string, filename to write the produced .obj mesh
        @label_path: string filename to write the produced .mtl file
        @json_param_path: a json file containing the parameter margins, in order to
                        to randomize the bridge structure
        !!! The wing walls initial placement is 90 degrees with respect to the
            abutment walls (piedroit) and facing towards the west.
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
        self.json_data_path = json_data_path
        self.obj_path=obj_path
        self.label_path=label_path
        self.json_param_path = json_param_path
        with open(json_data_path) as json_file:
            self.data = json.load(json_file)
        self.initialize_visibility()
        if self.json_param_path is not None:
            with open(self.json_param_path) as json_file:
                self.params = json.load(json_file)
            self.randomize_data()
        self.set_building_blocks()
        self.initialize_mesh_data()


    def initialize_visibility(self, value=True):
        """
        sets all building blocks to the same value
        """
        for item in self.data:
            item['Visible'] = value


    def dependent_property_no_bounds(self, rule):
        """
        this function calculates the random value to be assigned to a dimension
        or angle property, based on the rule defined in the  entry of
        the json file defining the parameters dependency.
        This function does not take into account the min and max bounds.
        INPUTS:
        @rule : dictionary object retrieved from the json file, denoting the
                independent variable and the Coefficient
        OUTPUTS:
        @new_value: the value of the property if it is dependent on another
                    property, None otherwise
        """
        if rule['Coefficient'] is None:
            return None
        assert len(rule['Coefficient'])==len(rule['Dependance']), 'the number of coefficients should be the same as that of independent variables'
        new_value = 0
        for i, var in enumerate(rule['Dependance']):
            mother_block = [b for b in self.data if b['Label']==var['Label']]
            if 'Nom' in var.keys():
                mother_block = [b for b in mother_block if b['Nom']==var['Nom']]
            if len(mother_block)==0: continue
            independent_value = mother_block[0]
            for d in var['Contrainte']:
                independent_value = independent_value[d]
            if rule['Coefficient'][i]['Fix']:
                coef = rule['Coefficient'][i]['Valeur']
            else:
                coef = random.uniform(
                            rule['Coefficient'][i]['Valeur'][0],
                            rule['Coefficient'][i]['Valeur'][1])
            new_value += coef*independent_value
        if "Constant" in rule.keys():
            new_value += rule["Constant"]

        return new_value


    def calculate_min_max(self, rule):
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
        if 'Min' in rule.keys():
            min_ = rule['Min']
        if 'Max' in rule.keys():
            max_ = rule['Max']
        if 'rMin' in rule.keys():
            rmin = self.dependent_property_no_bounds(rule['rMin'])
            if rmin is None:
                pass
            else:
                min_ = max(min_, rmin)
        if 'rMax' in rule.keys():
            rmax = self.dependent_property_no_bounds(rule['rMax'])
            if rmax is None:
                pass
            else:
                max_ = min(max_, rmax)

        return min_, max_


    def randomize_data(self):
        """
        this function uses the parameter margins to randomize the bridge
        dimensions
        """
        for element in self.params:
            affected_data = [i for i, x in enumerate(self.data) if x['Label']==element['Label']]
            if 'Nom' in element.keys():
                affected_data = [i for i in affected_data if self.data[i]['Nom']==element['Nom']]
            if len(affected_data)==0: continue
            # randomly select which blocks will be visible
            for index in affected_data:
                if not element['Necessaire'] and np.random.choice([True, False], p=[0.2, 0.8]):
                    self.data[index]['Visible'] = False
                    continue
            # calculate a random value for each element based on the rules
            for property, property_content in element.items(): # property should be dimensions or angle
                if property in ['Label', 'Nom', 'Necessaire']:
                    continue
                for specific_property, rules in property_content.items(): # specific property should be largeur, longueur, hauteur for dimensions, heading, roll or tilt for angle
                    for rule in rules:
                        min_, max_ = self.calculate_min_max(rule)
                        # here we could potentially add a pdf implementation
                        if rule['Coefficient'] is None:
                            if 'MaxProbability' in rule.keys():
                                new_value = np.random.choice([max_, random.uniform(min_, max_)],
                                             p=[rule['MaxProbability'], 1-rule['MaxProbability']])
                            else:
                                new_value = random.uniform(min_, max_)
                        else:
                            new_value = self.dependent_property_no_bounds(rule)
                            new_value = np.clip(new_value, min_, max_)
                        if new_value is None: continue
                        if new_value==-math.inf or new_value==math.inf:
                            warnings.warn('\n##-----## \nWarning: {} of {} is set' \
                                        'to infinity for element {}. This value will be '\
                                        'ignored\n##-----##'.format(
                                        property, specific_property, self.data[index]['Nom']))
                            pdb.set_trace()
                            continue
                        for index in affected_data:
                            for more_specific_property in rule['Affectation']: # more specific property should be D1, D2, Dec or V
                                self.data[index][property][specific_property][more_specific_property] = new_value


    def initialize_mesh_data(self):
        self.mesh ='mtllib ./Label.mtl\n'
        self.normals=''
        self.texture ='vt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n'
        self.faces=''
        self.materials ='newmtl material_A\nd 0.5\n'


    def get_dimension_parent(self, index, dimension):
        """
        returns the index of the building block whose given dimension
        should be used
        """
        i = index
        # find dependency for dimension measurements
        while self.data[i]['Dimensions'][dimension]['Contrainte']>=0:
            if self.data[i]['Dimensions'][dimension]['Contrainte']==i:
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['Dimensions'][dimension]['Contrainte']
        return i


    def get_tilt_parent(self, index, dimension):
        """
        returns the index of the building block whose tilt for the given dimension
        should be used
        """
        i = index
        while self.data[i]['Dimensions'][dimension]['ContrainteDec']>=0:
            if self.data[i]['Dimensions'][dimension]['ContrainteDec']==i:
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['Dimensions'][dimension]['ContrainteDec']
        return i


    def get_rotation_parent(self, index, rot_axis):
        """
        returns the index of the building block whose rotation angle for the given
        axis should be used
        """
        i = index
        while self.data[i]['Angle'][rot_axis]['Contrainte']>=0:
            if self.data[i]['Angle'][rot_axis]['Contrainte']==i:
                # normally we should never go in here
                sys.exit('Element constraint points to itself, infinite loop alert!')
            else:
                i = self.data[i]['Angle'][rot_axis]['Contrainte']
        return i


    def get_dimensions(self, index):
        """
        This function returns the dimensions of the cube and the offset(decalage),
        taking into account the constraints
        INPUTS:
        @index: index of the building block whose rotation angle we want to
                find
        dimension: string, indication the dimension, in {'Longueur', 'Largeur', 'Hauteur'}
        OTPUTS:
        dim1: first dimension (towards positive face)
        dim2: second dimension (towards negative face)
        tilt: offset, allowing the cube to tilt
        """
        dimensions = dict()
        for dimension in self.data[index]['Dimensions'].keys():
            i_parent = self.get_dimension_parent(index, dimension)
            dim1 = self.data[i_parent]['Dimensions'][dimension]['D1']
            dim2 = self.data[i_parent]['Dimensions'][dimension]['D2']

            # find dependency for tilt (decalage)
            i_parent_tilt = self.get_tilt_parent(index, dimension)
            tilt = self.data[i_parent_tilt]['Dimensions'][dimension]['Dec']

            dimensions[dimension] = (dim1, dim2, tilt)

        # width, length, height
        return dimensions['Longueur'], dimensions['Largeur'], dimensions['Hauteur']


    def get_rotation_angles(self, index):
        """
        This function returns the rotation angle in degrees, taking into
        account the constraints
        rot_axis: string, indication of the rotation axis
        If the obj is loaded with Y forward, Z up, this corresponds to:
                z: 'Heading' (bridge height - hauteur)
                y: 'Tilt' (bridge length - largeur)
                x: 'Roll' (bridge width - longueur)
        INPUTS:
        @index: index of the building block
        OUTPUTS:
        @ anglet, tilt, roll : corresponding angles in degrees
        """
        angles = dict()
        # i = index
        for rot_axis in self.data[index]['Angle'].keys():
            i_parent = self.get_rotation_parent(index, rot_axis)
            angles[rot_axis] = self.data[i_parent]['Angle'][rot_axis]['V']

        # return x, y, z axis rotation angles in degrees
        return angles['Roll'], angles['Tilt'], angles['Heading']


    def get_translation_offset(self, index):
        """
        This function calculates the new center of a child building block
        (assuming the old center is (0,0,0)), so that a child's node is connected
        to a parent's node. The information on the parent building block, and the
        connecting nodes are given in the JSON file.
        INPUTS:
        @index: child building block's index
        OUTPUTS:
        @center_point_coords: numpy array containing the x,y,z coordinates where
            the child building block should be centered
        """
        offset_coords = np.zeros(3)
        # if this building block has no constraints, offset is (0,0,0)
        if (self.data[index]['Contrainte']['Element']<0 or
            self.data[index]['Contrainte']['N1']<0 or
            self.data[index]['Contrainte']['N2']<0):
            return offset_coords

        while self.data[index]['Contrainte']['Element']>=0:
            # index of parent building block
            i_parent=self.data[index]['Contrainte']['Element']
            # assert self.data[index]['Contrainte']['N1']>=0, "Undefined parent node in the constraint declaration"
            # assert self.data[index]['Contrainte']['N2']>=0, "Undefined child node in the constraint declaration"

            if self.data[index]['Contrainte']['N1']>=0:
                # node (vertex) of parent building_block
                parent_node = self.data[index]['Contrainte']['N1']
                parent_node_coords = self.building_blocks[i_parent].global_coords[parent_node]

            if self.data[index]['Contrainte']['N2']>=0:
                # node (vertex) of child building block
                child_node = self.data[index]['Contrainte']['N2']
                child_node_coords = self.building_blocks[index].global_coords[child_node]

            # define the new center of the current building block, so that
            # the node1 (N1) of the parent block and the node2 (N2) of the
            # child block coincide
            offset_coords += parent_node_coords - child_node_coords

            index = i_parent

        return offset_coords


    def set_building_blocks(self):
        """
        This function creates a list of hexahedron objects describing the bridge
        building blocks
        """
        for index, building_block in enumerate(self.data):
            width, length, height = self.get_dimensions(index)
            rotx, roty, rotz = self.get_rotation_angles(index)
            self.building_blocks.append(Hexahedron(index, building_block['Nom'],
                                        width, length, height, rotx, roty, rotz))
            # remove coordinates if block is not visible
            if not self.data[index]['Visible']:
                self.building_blocks[index].global_coords = None
                continue
            offset = self.get_translation_offset(index)
            self.building_blocks[index].global_coords += offset


    def create_mesh(self):
        """
        This function processes the JSON file and creates the obj mesh data
        """
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
        assert len(self.building_blocks)==len(self.data), "seems that there is not enough hexahedron objects"
        for index, building_block in enumerate(self.data):
            if building_block['Visible']:
                vis_index+=1
            else:
                continue
            # add material label if not already there
            label = f'Label_{building_block["Label"]}_Name_{building_block["Nom"]}'
            if (label not in labels):
                labels.append(label)
                # self.materials += 'newmtl material_{}\nmap_kd Label/{}.png\n'.format(building_block['Label'], building_block['Label'])
                self.materials += f'newmtl material_{label}\n'
                self.faces += f'\n\n\nusemtl material_{label}\n'

            # create building block faces and add to mesh
            for i, face in enumerate(face_desc):
                face_coords = self.building_blocks[index].global_coords[face_nodes[face]]
                normal = alg.get_normal(face_coords)

                if face in ['front', 'back']:
                    self.mesh += 'v {} {} {}\nv {} {} {}\nv {} {} {}\nv {} {} {}\n'.format(*face_coords.flatten())
                self.normals += 'vn {} {} {}\n'.format(*normal)
                self.faces += 'f {}/1/{} {}/2/{} {}/3/{} {}/4/{}\n'.format(vis_index*8+face_vertices[face][0], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][1], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][2], vis_index*6+i+1,
                                                                           vis_index*8+face_vertices[face][3], vis_index*6+i+1)
                # self.faces += 'f {}/1 {}/2 {}/3 {}/4\n'.format(*(index*8+face_vertices[face]))

        self.mesh += self.normals + self.texture + self.faces


    def save_mesh(self):
        """
        saves mesh with material details as an .obj file
        """
        with open(self.label_path, "w+") as f:
            f.write(self.materials)
        with open(self.obj_path, "w+") as f:
            f.write(self.mesh)
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
