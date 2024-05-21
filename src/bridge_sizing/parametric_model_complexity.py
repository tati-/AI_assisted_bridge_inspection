"""
This script calculates how many independent and how many dependent variables
are to be defined for the bridge dimensioning, based on the parametric JSON model.
It is not a crucial script that is used in the pipeline, it is just for
planning and comprehension pursposes
"""
import os
import re
import sys
import pdb
import json
import flatdict
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-basefile', help='input file path [acceptable formats: .json]', type=str, required=True)
    arg('-params', help='json file path with parameters', type=str)
    args = parser.parse_args()
    return args


def json_complexity(json_basefile):
    """
    calculates, for a base json file, how many parameters in total are defined
    (dimensions, angles etc), and how many of them are independent
    """
    with open(json_basefile) as f:
        data = json.load(f)
    n_total, n_indep = 0, 0
    pattern_params = '^Dimensions\.(Longueur|Largeur|Hauteur)\.D|Angle\.(Heading|Roll|Tilt)\.V'
    pattern_constraints_single = '^Dimensions\.(Longueur|Largeur|Hauteur)\.ContrainteDec|Angle\.(Heading|Roll|Tilt)\.Contrainte'
    pattern_constraints_double = '^Dimensions\.(Longueur|Largeur|Hauteur)\.Contrainte$'
    for block in data:
         block_flat = flatdict.FlatDict(block, delimiter='.')
         n_total += len([key for key in block_flat.keys() if re.search(pattern_params, key)])
         n_indep += len([key for key,val in block_flat.items()
                                    if re.search(pattern_constraints_single, key)
                                    and val==-1])
         n_indep += 2*len([key for key,val in block_flat.items()
                                    if re.search(pattern_constraints_double, key)
                                    and val==-1])

    return n_total, n_indep


if __name__ == "__main__":
    args = get_args()
    n_total, n_indep = json_complexity(args.basefile)
    print(f'{args.basefile} contains in total {n_total} parameters, but {n_indep} of them' \
    f' are independent.')
    """
    ############################################################################
                                    END
    ############################################################################
    """
