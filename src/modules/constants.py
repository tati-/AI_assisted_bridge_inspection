import os
import sys

# CLASSES_PIPO = {'background':0,
#                 'piedroit':1,
#                 'traverse':2,
#                 'mur':3,
#                 'gousset':4,
#                 'corniche':5}

WIDTH = 640
HEIGHT = 480

CLASSES_PIPO = {'background':0,
                'abutment':1,
                'deck':2,
                'wing_wall':3,
                'haunch':4,
                'edge_beam':5}

# LABELS_PIPO = ['background', 'piedroit', 'traverse', 'mur', 'gousset', 'corniche']

LABELS_PIPO = ['background', 'abutment', 'deck', 'wing_wall', 'haunch', 'edge_beam']

# class_weights_pipo = {'background':0.05*6,
#                 'piedroit':0.2*6,
#                 'traverse':0.2*6,
#                 'corniche':0.5*6,
#                 'gousset':0.2*6,
#                 'mur':0.2*6}

CLASS_WEIGHTS = {'background':0.05*6,
                'abutment':0.2*6,
                'deck':0.2*6,
                'edge_beam':0.5*6,
                'haunch':0.2*6,
                'wing_wall':0.2*6}
