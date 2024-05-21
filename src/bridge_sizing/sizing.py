"""
Bridge sizing script
This script infers the optimal geometric parameters to describe a bridge,
based on a labeled point cloud and a file containing information on the bridge's
structure.
It produces a linear dictionary with parameters containing the bridge's dimensions
Unless otherwise specified, the dictionary is saved in JSON format in the same
directory as the script
example usage: python sizing.py -pc ./json/parameters.json
               python sizing.py @config.txt

THIS SCRIPT IS INCOMPLETE, IT IS A WORK IN PROGRESS
NOTE: for the moment only the deck parameters are optimized and inferred
NOTE2: For the demo to CEREMA, since the optimization function to automatically
      define the parameters is not yet finalized, we will use an already
      sized bridge, whose dimensions we will pass via the basefile (instead of
      just a skeleton, the basefile will be a fully dimensioned bridge for the
      demo). For the automatic minimization the following things should change:
      1. pass a skeleton JSON via the -b argument, not a fully dimensioned bridge.
      2. Uncomment the part of the code that actually infers the parameters
"""
import os
import re
import sys
import json
import random
import argparse
import numpy as np
import numpy.typing as npt
from pathlib import Path
from scipy.optimize import minimize, show_options, Bounds

sys.path.append('..')
import modules.visualization as vis
import modules.point_cloud_utils as pcl
from modules.constants import CLASSES_PIPO, LABELS_PIPO
import modules.optimization_utils as optim
from classes.bridge import BridgeModel as Bridge

# sys.stderr = open('error.log', 'w')

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-pc', help='path to point cloud [accepted formats: GLB]', type=str, required=True)
    # comment or erase the following line for the real case scenario
    # arg('-b', help='skeleton json file for specific type of bridge', type=str, default='./json/pont_bron_chem_des_balmes.json')
    # uncomment the following for the real scenario
    arg('-b', help='skeleton json file for specific type of bridge [default: ./json/skeleton.json]', type=str, default='./json/skeleton.json')
    arg('-o', help='output parameter file directory [default: ./parameters.json]', type=str, default='./parameters.json')
    arg('-n', help='level of noise to introduce to the point cloud [default: 0]', type=float, default=0.0)
    args = parser.parse_args()
    return args


def initialize_bridge(glb_points: npt.ArrayLike,
                    glb_labels: npt.ArrayLike,
                    bridge: Bridge):
    """
    given a glb file containing a labeled point cloud, creates an initial bridge
    object approximating the geometry
    """
    for i in range(1, len(LABELS_PIPO)): # not interested in label 0
        points = glb_points[glb_labels==i, :]
        if len(points)<5:
            continue
        # find bridge blocks that could correspond
        candidates = [(j, bblock) for j,bblock in enumerate(bridge.building_blocks)
                        if bblock.label==LABELS_PIPO[i]]
        for j, block in candidates:
            points_ = pcl.intralabel_point_filter(points, LABELS_PIPO[i], block.name)
            if len(points_)<5:
                block.visible = False
                continue
            block.plane = plane_model
            block.from_points(points_)
            bridge.update_child_blocks(parent_id=block.id)
            ###### DEBUG
            # nPoints = np.sum(bridge.building_blocks[j].contains(points_))
            # print(f'{len(points_)} points formed block {bridge.building_blocks[j].name}, that now has {nPoints}')
            ######

    # update before returning, otherwise the bridge data might not reflect the
    # building blocks' state
    bridge.update_data()
    return bridge


def size_bridge(pc_file: str,
                base_file: str,
                output_file: str,
                noise: float=0):
    """
    given a labeled point cloud find the optimal parameters to define a bridge's
    digital twin
    @noise: float in (0,1), noise to add to the point cloud
    """
    glb_points, glb_labels = pcl.load_pointCloud_glb(pc_file)
    # move points to be centered around (0,0,0)
    glb_points -= np.mean(glb_points, axis=0)
    # remove background points from point cloud
    glb_points = glb_points[glb_labels!=0, ...]
    glb_labels = glb_labels[glb_labels!=0]
    # smooth point clould
    glb_labels = pcl.majority_filter(glb_points, glb_labels)

    # filter points so that some classes are confined to belong to planes
    glb_points, glb_labels = pcl.plane_filter(glb_points, glb_labels)
    vis.display_pc(glb_points, glb_labels)
    breakpoint()

    # introduce artificial noise to the point cloud
    glb_labels = pcl.add_noise(glb_labels, noise)

    ############################################################################
    #                          INITIALIZATION                                  #
    ############################################################################
    # INITIALIZATION OF BRIDGE STRUCTURE
    bridge = Bridge(base_file)

    bridge = initialize_bridge(glb_points, glb_labels, bridge)
    bounds = optim.bridge_bounds(bridge)
    # make sure bridge values respect the bounds
    bridge.from_fv(np.clip(bridge.fv, a_min=bounds.lb, a_max=bounds.ub))

    # only keep inlier points (as defined by the bridge initialization)
    glb_points = np.concatenate([block.points for block in bridge.building_blocks
                                            if block.visible], axis=0)
    glb_labels = np.concatenate([np.repeat(CLASSES_PIPO[block.label], len(block.points))
                                            for block in bridge.building_blocks
                                            if block.visible], axis=0)

    ############################################################################
    #                          INITIAL ALIGNMENENT                             #
    ############################################################################
    # align point cloud to bridge, otherwise optimization does not make
    # sense
    deck_points = glb_points[glb_labels==CLASSES_PIPO['traverse'], ...]

    bridge.align_to_point_cloud(deck_points)

    ############################################################################
    #                           MODEL OPTIMIZATION                             #
    ############################################################################
    # OPTIMIZATION OF DECK PARAMETERS
    deck = [bl for bl in bridge.building_blocks if bl.label=='traverse'][0]
    # for the mock case, artificially impose some wrong dimensions
    # deck.dimensions['width'] *= 0.8
    # deck.dimensions['length'] *= 1.1

    constraints = optim.deck_constraints(deck)
    bounds = optim.block_bounds(deck)
    deck.from_fv(np.clip(deck.fv, a_min=bounds.lb, a_max=bounds.ub))

    # OPTIMIZE THE ENTIRE BLOCK FV
    init = optim.point2mesh_ssd(deck.fv, deck)
    init2 = optim.iou_2D(deck.fv, deck)
    init3 = optim.test(deck.fv, deck)
    # vis.display_pc(glb_points, glb_labels, vertices=deck.global_coords)
    # COBYLA does not handle bounds, so they need to be passed
    # as constraints if wanted
    method = 'trust-constr' # COBYLA, trust-constr, SLSQP
    # show_options(solver='minimize', method=method)
    res = minimize(optim.test, deck.fv, method=method,
                    args=(deck),
                    bounds=bounds,
                    constraints=constraints,
                    callback=optim.obj_func_monitor,
                    options={'maxiter': 1000, 'disp': True})
    ###
    import matplotlib.pyplot as plt
    plt.plot(optim.obj_value)
    plt.show()
    breakpoint()
    ###
    bridge_fv = bridge.fv
    bridge_fv[deck.id: deck.id+len(deck.fv)] = res.x
    bridge.from_fv(bridge_fv)
    vis.display_pc(glb_points, glb_labels, vertices=deck.global_coords)
    breakpoint()

    ############################################################################
    #          UPDATE AND SAVE OPTIMAL BRIDGE PARAMETERS                       #
    ############################################################################
    bridge.to_blender(Path(pc_file).with_stem(f'{Path(pc_file).stem}_after_optim_{method}').with_suffix('.blend'))
    with open(output_file, 'w') as f:
        json.dump(bridge.params, f, indent=2)

    if np.any([x is None for x in bridge.params.values()]):
        sys.stderr.write('Automatic sizing failed\n')
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    if Path(args.pc).suffix!='.glb':
        sys.exit('Point cloud should be in GLB format!')
    if Path(args.b).suffix!='.json':
        sys.exit('Base structure file should be of JSON format!')
    if Path(args.o).suffix!='.json':
        sys.exit('Output file should be of JSON format!')

    size_bridge(args.pc, args.b, args.o)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
