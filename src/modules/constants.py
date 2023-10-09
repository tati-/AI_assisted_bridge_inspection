import os
import sys

CLASSES_PIPO = {'background':0,
                'piedroit':1,
                'traverse':2,
                'mur':3,
                'gousset':4,
                'corniche':5}

LABELS_PIPO = ['background', 'piedroit', 'traverse', 'mur', 'gousset', 'corniche']

class_weights_pipo = {'background':0.05*6,
                'piedroit':0.2*6,
                'traverse':0.2*6,
                'corniche':0.5*6,
                'gousset':0.2*6,
                'mur':0.2*6}
