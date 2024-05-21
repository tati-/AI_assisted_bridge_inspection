import os
import sys
import pdb
import math
import random
import warnings
import numpy as np
import numpy.typing as npt
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


def orthogonal(v0: npt.ArrayLike, v1: npt.ArrayLike, v2: npt.ArrayLike, tol: float=0.01) -> bool:
    """
    check if 3 vectors are orthogonal, by examining whether the cross product of the
    two first is paraller to the third
    """
    v2prime = np.cross(v0, v1)
    if math.isclose(np.linalg.norm(np.cross(v2prime, v2)), 0, abs_tol=tol):
        return True
    else:
        return False


def sample_positions(init_location, init_rotation, location_ranges, rotation_ranges, n_samples):
    """
    samples random location coordinates and rotation angles for each axis
    INPUTS:
    @init_location: array length 3, with xyz coordinates
    @init_rotation: array length 3, with rotation around x, y, and z, in degrees
    @location_ranges: array length 3x2 absolute margins on each axis (in meters)
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


def extend_3Dline_along_y(point1: npt.ArrayLike, point2: npt.ArrayLike, extension):
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


def rotate_points(coord: npt.ArrayLike, rot_angles: npt.ArrayLike) -> npt.ArrayLike:
    """
    rotates points based on the rotation angles per axis
    (active counterclockwise extrinsic rotation)
    INPUTS:
    @coord : array with point coordinates (3 x nPoints)
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
    try:
        return r@coord
    except Exception as error:
        print(f'Rotation was not succesful: {error}')


def get_normal(points: npt.ArrayLike) -> npt.ArrayLike:
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

def point_projected_on_plane(plane_points: npt.ArrayLike, point: npt.ArrayLike) -> npt.ArrayLike:
    """
    returns the 3d coordinates of the projection of a point on a 3d plane. The plane
    is defined by 3 or more points.
    """
    normal = get_normal(plane_points)
    plane = Plane(point=plane_points[0], normal=normal)
    point_projected = plane.project_point(Point(point))

    return point_projected


def area_from_points(points: npt.ArrayLike) -> float:
    """
    return the area enclosing a number of points.
    INPUT:
    @points: nPoints x 3 array of point coordinates
    """
    try:
        area = ConvexHull(points).area
    except Exception as e:
        area = 0
        warnings.warn(str(e))

    return area


def volume_from_points(points: npt.ArrayLike) -> float:
    """
    return the volume enclosing a number of points. Returns 0
    if the points cannot form a 3d volume
    INPUT:
    @points: nPoints x 3 array of point coordinates
    """
    try:
        volume = ConvexHull(points).volume
    except Exception as e:
        volume = 0
        warnings.warn(str(e))

    return volume


def points_in_mesh(mesh_points: npt.ArrayLike, test_points: npt.ArrayLike) -> npt.ArrayLike:
    """
    checks if a point lies inside the 3d polygon defined by points
    INPUTS:
    @mesh_points: the points that define the n_dimensional mesh
    @test_points: points to be tested on whether they lie inside the mesh
    OUTPUT:
    boolean array
    """
    assert mesh_points.shape[-1]==test_points.shape[-1], 'Points defining the polygon and new point should have the same dimension!'
    return Delaunay(mesh_points).find_simplex(test_points) >= 0


def initialize_tilt(points: npt.ArrayLike) -> npt.ArrayLike:
    """
    Infer width, length and height rough tilt values, based on the points
    distribution
    width tilt: x axis, offset between width on positive and negative z
    length tilt: y axis, offset between length on positive and negative x
    height tilt: z axis, offset between length on positive and negative x
    """
    # width offset
    med_pos = np.mean(points[points[:,2]>0], axis=0)
    med_neg = np.mean(points[points[:,2]<0], axis=0)
    width_tilt = med_pos[0] - med_neg[0]
    # length offset
    med_pos = np.mean(points[points[:,0]>0], axis=0)
    med_neg = np.mean(points[points[:,0]<0], axis=0)
    #### DEBUG
    # vis.plot_3d_point_cloud(points, np.vstack([med_pos, med_neg]))
    ####
    length_tilt = med_pos[1] - med_neg[1]
    # height offset
    height_tilt = med_pos[2] - med_neg[2]
    return np.array([width_tilt, length_tilt, height_tilt])


def point2mesh_distance(points: npt.ArrayLike, mesh: o3d.geometry.TriangleMesh) -> npt.ArrayLike:
    """
    this function calculates the minimum distance of a set of points to a triangular mesh
    (http://www.open3d.org/docs/latest/tutorial/geometry/distance_queries.html)
    INPUTS:
    @points : nPoints x 3 array with the coordinates of the query points
    @mesh: an open3d geometry object representing a triangle mesh
    OUTPUT:
    @unsigned_distance: nPoints length array with the minimum unsigned distance
                        of each query point from the mesh
    """
    trimesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(trimesh)
    # make sure the query points have the right format
    points = points.astype(np.float32)
    points = points[np.newaxis, ...] if points.ndim == 1 else points
    # query_point = o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32)
    # Compute distance of the query point from the surface
    # While the unsigned distance can always be computed, the signed distance and
    # the occupancy are only valid if the mesh is watertight and the inside and
    # outside are clearly defined. The signed distance is negative if the query
    # point is inside the mesh. The occupancy is either 0 for points outside the
    # mesh and 1 for points inside the mesh.
    unsigned_distance = scene.compute_distance(points)
    # signed_distance = scene.compute_signed_distance(query_point)
    # occupancy = scene.compute_occupancy(query_point)

    return unsigned_distance.numpy()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# faster than delauney for one point, but not for more
# def pnt_in_pointcloud(points: npt.ArrayLike, new_pt: npt.ArrayLike) -> bool:
#     """
#     checks if a point lies inside the 3d polygon defined by points
#     """
#     assert points.shape[-1]==new_pt.shape[-1], 'Points defining the polygon and new point should have the same dimension!'
#     hull = ConvexHull(points)
#     new_hull = ConvexHull(np.concatenate((points, [new_pt])))
#     # return np.array_equal(new_hull.vertices, hull.vertices)
#     return np.array_equal(new_hull.simplices, hull.simplices)
