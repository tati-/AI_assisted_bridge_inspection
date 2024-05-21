"""
IFC Builder script
This script generates an IFC model of a bridge. It takes as input 2 json files,
one with the structure and the dependencies of a bridge (specific to the bridge type)
and one with a linear dictionary with parameters containing the bridge's dimensions
Unless otherwise specified, the IFC file is saved in the same directory as the
script
example usage: python ifc_builder.py -i ./json/parameters.json
               python ifc_builder.py @config_ifc_builder.txt
"""
import os
import re
import sys
import pdb
import json
import argparse
import numpy as np
from pathlib import Path

from classes.bridge import BridgeModel as Bridge


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-i', help='input json file with parameter dictionary', type=str, required=True)
    arg('-b', help='skeleton json file for specific type of bridge [default: ./json/skeleton.json]', type=str, default='./skeleton.json')
    arg('-o', help='output ifc file directory [default: ./bridge.ifc]', type=str, default='./bridge.ifc')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    if Path(args.i).suffix!='.json':
        sys.exit('Input parameter file should be of JSON format!')
    if Path(args.b).suffix!='.json':
        sys.exit('Base structure file should be of JSON format!')
    if Path(args.o).suffix!='.ifc':
        sys.exit('Output file should be of IFC format!')

    bridge = Bridge(args.b)
    with open(args.i) as f:
        params = json.load(f)
    bridge.from_params(params)
    bridge.characterize_wing_walls()

    # create the directory to the output file if it does not exist
    os.makedirs(Path(args.o).parent, exist_ok=True)
    bridge.to_ifc(ifcPath=args.o)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    ## temp
    # bridge = br.BridgeModel(args.basefile)
    # dictio = bridge.to_params()
    # with open(Path(args.basefile).parent.joinpath('parameters.json'), 'w') as f:
    #     json.dump(dictio, f, indent=2)
    # with open(Path(args.basefile).parent.joinpath('params.json'), 'w') as f:
    #     json.dump(list(bridge.to_fv()), f, indent=2)
    # pdb.set_trace()
    ##
