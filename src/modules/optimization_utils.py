import os
import sys
import pdb
import math
import copy
import warnings
import numpy as np
import open3d as o3d
import numpy.typing as npt
from itertools import product
from scipy.optimize import Bounds, OptimizeResult, LinearConstraint
from scipy.optimize import minimize, brute, differential_evolution

import classes.bridge as br
import classes.building_block as bbl
from .decorators import timer
import modules.algebra_utils as alg
from modules.constants import LABELS_PIPO

obj_value = list()


def obj_fun_alignment(offset: npt.ArrayLike, points_pc: npt.ArrayLike,
                    points_mesh: npt.ArrayLike) -> float:
    """
    objective function to make a set of point cloud points fit as much as possible
    to the edge points of a mesh.
    The parameter vector to find is to be applied to the mesh edge points(vertices).
    INPUTS:
    @offset: 1x3 vector, with the vertices offset in the x,y,z axes
    @points_pc: nx3 array, containing a list of point cloud coordinates
    @points_mesh: mx3 array, containing a list of vertices that define a volume,
                inside which the points should fit as much as possible
    """
    points_mesh = points_mesh + offset
    return 1-np.sum(alg.points_in_mesh(points_mesh, points_pc))/len(points_pc)


def align_model_to_points(points_pc: npt.ArrayLike, points_model: npt.ArrayLike,
                            method: str=None) -> npt.ArrayLike:
    """
    aligns point belonging to a point cloud to edge points of one or more hexahedra
    belonging to a bridge model. This is done by finding the offset that maximises
    the number of points contained in the hexahedra
    INPUTS:
    @points_pc: nx3 array, containing a list of point cloud coordinates
    @points_model: mx3 array, containing a list of vertices that define a volume,
                inside which the points should fit as much as possible
    @method: otpimization method
    """
    init_offset = np.zeros(3)
    # local optimization version
    # res = minimize(obj_fun_alignment, init_offset, method=method,
    #                 args=(points_pc, points_model),
    #                 callback=obj_func_monitor,
    #                 options={'disp': True})
    # import matplotlib.pyplot as plt
    # plt.plot(obj_value)
    # plt.show()

    # global optimization version
    # infer ranges for global minimization from the range of the point cloud
    mins = np.min(points_pc, axis=0)-np.min(points_model, axis=0)
    maxs = np.max(points_pc, axis=0)-np.max(points_model, axis=0)
    ranges = ((mins[0], maxs[0]), (mins[1], maxs[1]), (mins[2], maxs[2]))
    res = brute(obj_fun_alignment, ranges=ranges,
                    args=(points_pc, points_model),
                    Ns=20,
                    # finish=None,
                    full_output=True)
    # res = differential_evolution(obj_fun_alignment, bounds=ranges,
    #                 args=(points_pc, points_model),
    #                 disp=True)

    # return optimization result
    return res


def points_in_blocks(points: npt.ArrayLike, bridge: br.BridgeModel) -> npt.ArrayLike:
    """
    returns a boolean matrix of size nPoints x nHexahedra, indicating whether
    a point lies in the interior of an hexahedron
    INPUTS:
    @points: nPoints x 3 numpy array with point coordinates
    @bridge: bridge object
    OuTPUTS:
    @point_presence: boolean array, size nPoints x Nhexahedra, where element
                        [i,j] indicates if point i lies into the hexahedron j
    """
    point_presence = [block.contains(points) for block in bridge.building_blocks]

    return np.asarray(point_presence).T


def n_points_objective_function(fv: npt.ArrayLike, bridge: br.BridgeModel,
                                points: npt.ArrayLike, point_labels: npt.ArrayLike,
                                steps: int=1) -> float:
    """
    Objective function, normalized sum of #points falling in the hexahedron of their class,
    minus #points falling in the hexahedron of another class
    INPUTS:
    @fv: feature vector with bridge parameters to optimize
    @bridge: bridge object
    @points: nPoints x 3 numpy array with point coordinates
    @point_labels: nPoints x 1 numpy array with an integer label for each point
    @steps: how many points to skip before taking the next sample
    """
    bridge.from_fv(fv)
    points = points[::steps, :3]
    point_labels = point_labels[::steps]
    # get all labels that appear in the point cloud, except the background(0)
    labels_present = list(set(point_labels)).remove(0)
    labels_present = [l for l in labels_present if l<len(LABELS_PIPO)]

    sum = 0
    point_presence = points_in_blocks(points, bridge)
    for i,j in product(range(1, labels_present), repeat=2):
        points_i = points[point_labels==i, :3]
        hexahedra_j = [ind for ind,x in enumerate(bridge.building_blocks) if x.label==LABELS_PIPO[j]
                                        and x.is_complete()]
        points_in_block_classj = np.logical_or.reduce(point_presence[:, hexahedra_j], axis=1)
        points_classi_in_block_classj = points_in_block_classj[point_labels==i]
        # sum += np.select([i==j, i!=j], [1, -1])*np.sum(point_presence[:, hexahedra_j])/len(points_i)
        sum += np.select([i==j, i!=j], [1, -1])*np.sum(points_classi_in_block_classj)/np.sum(point_labels==i)

    return 1-sum/(len(labels_present)-1)**2


def test(fv: npt.ArrayLike, block: bbl.Hexahedron, steps: int=1) -> float:
    return np.sum(fv)


def point2mesh_ssd(fv: npt.ArrayLike, block: bbl.Hexahedron, steps: int=1) -> float:
    """
    Objective function, calculating the distance of each point of a point cloud
    from a model
    INPUTS:
    @fv: a feature vector with the block dimensions
    @block: the hexahedron object representing the building block
    @steps: the step of points to be used from the point cloud for the calculation
            of the objective function
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    distances = alg.point2mesh_distance(points, blo.triangle_mesh)

    # # return root mean square distance
    # return np.sqrt(np.sum(np.square(distances))/len(distances))
    # return sum of square distances
    return np.sum(np.square(distances))


def iou_2D(fv: npt.ArrayLike, block: bbl.Hexahedron, steps=1) -> float:
    """
    calculates the 2D intersection over union of the block points and the
    block vertices as area_intersection/area_union. We use the area and not
    the volume because more often than not the point cloud points form a planar surface,
    meaning the volume of the intersection is close to 0.
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    # create a triangle mesh for a single surface of the block
    # vertices = o3d.utility.Vector3dVector(blo.global_coords)
    # triangles = o3d.utility.Vector3iVector(blo.triangle_mesh.triangles[4:6, ...])
    # plane = o3d.geometry.TriangleMesh(vertices=vertices, triangles=triangles)
    # sample points on the block's surface
    # block_pc = plane.sample_points_uniformly(number_of_points=len(points))
    # block_points = np.asarray(block_pc.points)
    intersection_points = points[blo.contains(points), ...]
    intersection_area = alg.area_from_points(intersection_points)
    union_area = alg.area_from_points(points) + \
                 alg.area_from_points(blo.global_coords) - \
                 intersection_area
    # print(intersection_points.shape, points.shape)
    try:
        iou = intersection_area / union_area
    except:
        warnings.warn('IoU could not be calculated.')
        iou = 0.0

    return 1-iou


def iou_3D(fv: npt.ArrayLike, block: bbl.Hexahedron, steps=1) -> float:
    """
    calculates the 3D intersection over union of the block points and the
    block vertices as volume_intersection/volume_union
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    intersection_points = points[blo.contains(points), ...]
    union_points = np.vstack([points, blo.global_coords])
    # print(intersection_points.shape, points.shape)
    try:
        iou = alg.volume_from_points(intersection_points) / alg.volume_from_points(union_points)
    except:
        warnings.warn('IoU could not be calculated.')
        iou = 0.0

    return 1-iou


def volume_diff(fv: npt.ArrayLike, block: bbl.Hexahedron, steps: int=1) -> float:
    """
    Objective function, difference of volume of block points and block vertices
    The labels are NOT taken into account
    INPUTS:
    @fv: feature vector with bridge parameters to optimize
    @bridge: bridge object
    @points: nPoints x 3 numpy array with point coordinates
    @point_labels: nPoints x 1 numpy array with an integer label for each point
    @steps: how many points to skip before taking the next sample
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    if not blo.visible:
        return 0
    try: # if a block is not complete we do not want its zero volume to bias the loss
        return abs(alg.volume_from_points(points) - blo.volume)
        # ratio = alg.volume_from_points(blo.points)/blo.volume
    except:
        return 0


def mean_diff(fv: npt.ArrayLike, block: bbl.Hexahedron, steps: int=1) -> float:
    """
    Objective function, difference of point to block mean
    INPUTS:
    @fv: feature vector with bridge parameters to optimize
    @bridge: bridge object
    @points: nPoints x 3 numpy array with point coordinates
    @point_labels: nPoints x 1 numpy array with an integer label for each point
    @steps: how many points to skip before taking the next sample
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    if not blo.visible:
        return 0
    return np.mean(np.mean(points, axis=0) - np.mean(blo.global_coords, axis=0))


def ptp_diff(fv: npt.ArrayLike, block: bbl.Hexahedron, steps: int=1) -> float:
    """
    Objective function, difference of point to block peak to peak values
    INPUTS:
    @fv: feature vector with bridge parameters to optimize
    @bridge: bridge object
    @points: nPoints x 3 numpy array with point coordinates
    @point_labels: nPoints x 1 numpy array with an integer label for each point
    @steps: how many points to skip before taking the next sample
    """
    blo = copy.deepcopy(block)
    blo.from_fv(fv)
    points = blo.points[::steps, :3]
    if not blo.visible:
        return 0
    return np.mean(np.ptp(points, axis=0) - np.ptp(blo.global_coords, axis=0))


def obj_func_monitor(x, intermediate_result=None):
    global obj_value
    if intermediate_result is not None:
        # if intermediate_result.nit != intermediate_result.niter:
        #     pdb.set_trace()
        # print(f'nfev: {intermediate_result.nfev}')
        print(intermediate_result.x)
        # pdb.set_trace()
        obj_value.append(intermediate_result.fun)
    else:
        obj_value.append(None)


def block_bounds(block: bbl.Hexahedron) -> npt.ArrayLike:
    """
    get an array of absolute bounds for each feature
    # TODO: somehow read those bounds from the parameter json file,
    for now hard coded
    """
    lb = np.full(len(block.fv), fill_value=-np.inf)
    ub = np.full(len(block.fv), fill_value=np.inf)
    keep_feasible = np.full(len(block.fv), fill_value=False)
    if not block.visible:
        lb[...] = 0
        ub[...] = 0
        return Bounds(lb, ub, keep_feasible=keep_feasible)
    # d1 and d2 of dimensions cannot be negative
    lb[:2] = 0
    lb[3:5] = 0
    lb[6:8] = 0
    # no need to have angles outside the 360 range
    lb[9:] = -359
    ub[9:] = 359
    if block.label == 'traverse':
        # width d1 & d2
        lb[:2] = 2
        ub[:2] = 10
        # length d1 & d2 (p. 54 ponts cadres et portiques) # new
        lb[3:6] = 2
        ub[3:6] = 20
        # height d1 & d2 (fatness)
        lb[6:8] = 0.3
        ub[6:8] = 0.5
        # roll
        lb[9] = -2.5
        ub[9] = 2.5
        # tilt
        lb[10] = -5
        ub[10] = 5
    if block.label == 'piedroit':
        # width d1 d2 (this is not in the params file but seems logical - fatness)
        lb[:2] = 0.3
        ub[:2] = 0.5
        # height d1 d2
        lb[6:8] = 2.5
        ub[6:8] = 5
    if block.label == 'mur':
        # width d1 & d2
        lb[:2] = 1
        ub[:2] = 10
        # length d1 d2 (fatness)
        lb[3:5] = 0.3
        ub[3:5] = 0.5
        #height d1
        lb[6] = 0
        ub[6] = 5
        #height d2
        lb[7] = 1
        ub[7] = 5
        #height offset
        lb[8] = -5
        ub[8] = 5
    if block.label == 'corniche':
        # length d1 d2 (this is not in the params file but seems logical - fatness)
        lb[3:5] = 0.2
        ub[3:5] = 0.5
        # height d1 d2
        lb[6:8] = 0.3
        ub[6:8] = 0.5
    if block.label == 'gousset':
        #width d1 & d2
        lb[:2] = 0.2
        ub[:2] = 0.9
        # height d1 d2
        lb[6:8] = 0.2
        ub[6:8] = 0.3

    return Bounds(lb, ub, keep_feasible=keep_feasible)


def bridge_bounds(bridge: br.BridgeModel) -> npt.ArrayLike:
    """
    get an array of absolute bounds for each feature
    # TODO: somehow read those bounds from the parameter json file,
    for now hard coded
    """
    lb = np.full(len(bridge.fv), fill_value=-np.inf)
    ub = np.full(len(bridge.fv), fill_value=np.inf)
    keep_feasible = np.full(len(bridge.fv), fill_value=False)
    for ind, block in enumerate(bridge.building_blocks):
        i = ind*len(block.fv)
        bl_bounds = block_bounds(block)
        lb[i:i+len(block.fv)] = bl_bounds.lb
        ub[i:i+len(block.fv)] = bl_bounds.ub
        keep_feasible[i:i+len(block.fv)] = bl_bounds.keep_feasible

    return Bounds(lb, ub, keep_feasible=keep_feasible)


def deck_constraints(block: bbl.Hexahedron):
    """
    returns the equality and inequality constraints of the bridge deck (traverse)
    (constraints, not bounds)
    """
    fv = block.fv
    constraint_mat = np.zeros(len(fv))
    lb, ub = list(), list()
    # 1st constraint deck.height.d1 = 0.045*deck.width.d1
    constraint_mat[0] = 0.045
    constraint_mat[6] = -1
    lb.append(0)
    ub.append(0)
    # 2nd constraint deck.height.d2 = 0.045*deck.width.d1
    tmp = np.zeros(len(fv))
    tmp[0] = 0.045
    tmp[7] = -1
    constraint_mat = np.vstack((constraint_mat, tmp))
    lb.append(0)
    ub.append(0)
    # 3rd constraint deck.length.offset <= 0.5 * deck.width.d1
    tmp = np.zeros(len(fv))
    tmp[0] = 0.5
    tmp[5] = -1
    constraint_mat = np.vstack((constraint_mat, tmp))
    lb.append(0)
    ub.append(np.inf)
    # 4th constraint deck.length.offset >= -0.5 * deck.width.d1
    tmp = np.zeros(len(fv))
    tmp[0] = 0.5
    tmp[5] = 1
    constraint_mat = np.vstack((constraint_mat, tmp))
    lb.append(0)
    ub.append(np.inf)

    return LinearConstraint(constraint_mat, lb, ub, keep_feasible=True)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
