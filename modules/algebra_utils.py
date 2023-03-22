import os
import sys
import pdb
import math
import time
import random
import itertools
import numpy as np
from natsort import natsorted
from scipy import interpolate
from scipy.spatial.transform import Rotation as R


def collinear(point0, point1, point2, tol=0.01):
    """
    check if 3 points in 3d space are collinear by examining if the cross
    product of two vectors defined by them is 0
    """
    vector0 = np.asarray(point1) - np.asarray(point0)
    vector1 = np.asarray(point2) - np.asarray(point0)
    # print(np.linalg.norm(np.cross(vector0, vector1)))
    if math.isclose(np.linalg.norm(np.cross(vector0, vector1)), 0, abs_tol=tol):
        return True
    else:
        return False


def sample_positions(init_location, init_rotation, location_ranges, rotation_ranges, n_samples):
    """
    samples random location coordinates and rotation angles for each axis
    INPUTS:
    @init_location: array length 3, with xyz coordinates
    @init_rotation: array length 3, with rotation around x, y, and z, in degrees
    @location_ranges: array length 3 range of +- meters around the initial location
    @rotation_ranges: array length 3 range of +- degrees around the initial rotation
    @n_samples: integer, number of rotation versions to sample
    OUTPUTS:
    @rotations: matrix of size (n_samples)x3, with each row beeing a rotation
                version, in degrees
    """
    locations = np.empty((n_samples, 3))
    rotations = np.empty((n_samples, 3))
    rnd_gen = np.random.default_rng()

    for ax, marg in enumerate(location_ranges):
        # locations[:, ax] = rnd_gen.uniform(init_location[ax]-marg,
        #                                 init_location[ax]+marg, n_samples)
        locations[:, ax] = rnd_gen.uniform(marg[0], marg[1], n_samples)
    for ax, marg in enumerate(rotation_ranges):
        rotations[:, ax] = rnd_gen.integers(math.floor(init_rotation[ax])-marg,
                                        math.floor(init_rotation[ax])+marg, n_samples)

    return locations, rotations


def extend_3Dline_along_y(point1, point2, extension):
    """
    this function, given 2 points in 3d space, finds the 3d line that connects them
    it then proceeds in returning 2 new points, that belong to this line, but are further
    forward and backwards from the initial points, along the y axis.
    INPUTS
    @point1, point2 : arrays of size (3,), with the coordinates of 2 points in 3D space
    @extension: scalar value of the extension along the y axis
    """
    # define which point is front and which is back
    if point1[1]>point2[1]:
        front, back = point1, point2
    else:
        front, back = point2, point1

    x, y, z = [back[0], front[0]], [back[1], front[1]], [back[2], front[2]]

    f_line_x = interpolate.interp1d(y, x, fill_value='extrapolate')
    f_line_z = interpolate.interp1d(y, z, fill_value='extrapolate')

    new_y = front[1] + extension
    front_point = np.array([f_line_x(new_y), new_y, f_line_z(new_y)])
    new_y = back[1] - extension
    back_point = np.array([f_line_x(new_y), new_y, f_line_z(new_y)])

    return front_point, back_point


def rotate_point(coord, rot_angles):
    """
    rotates a point based on the rotation angles per axis (extrinsic rotation)
    INPUTS:
    @coord : array with point coordinates
    @rot_angles: array with counterclockwise rotation angles per axis, in degrees
    @rot_x (rot_angles[0]): pitch angle, 'Roll' in JSON file
    @rot_y (rot_angles[1]): roll angle, 'Tilt' in JSON file
    @rot_z (rot_angles[2]): yaw angle, 'Heading' in JSON file
    OUTPUTS:
    array with new coordinates
    """
    # create extrinsic rotation matrix from angles
    r = R.from_euler('xyz', rot_angles, degrees=True).as_matrix()
    # rotate point
    return r@coord


def get_normal(points):
    """
    This function calculates the normal of a plane surface that is
    created by the clockwise sequence of points
    INPUTS:
    @points: 4x3 array, each line being the local 3d coordinates of a point
    OUTPUTS:
    @normal: vector of length 3, face normal
    """
    n_points = points.shape[0]
    for i in range(n_points):
        normal = np.cross(points[(i+1)%n_points]-points[i],
                          points[(i-1)%n_points]-points[i])
        if np.any(normal):
            break

    return normal

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
