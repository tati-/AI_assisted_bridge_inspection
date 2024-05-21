import os
import re
import sys
import json
import flatdict

from .decorators import forall, verify_format


word2word = {'nom': 'name',
        'couleur': 'color',
        'longueur': 'width',
        'largeur': 'length',
        'hauteur': 'height',
        'dec' : 'offset',
        'contrainte': 'constraint',
        'contraintedec': 'offset_constraint',
        'angle': 'angles',
        'element': 'parent',
        'n1': 'parent_node',
        'n2': 'child_node',
        # specific for parameters file
        'necessaire': 'necessary',
        'affectation': 'dependent_var',
        'dependance': 'equation',
        'valeur': 'value',
        'fix': 'fixed',
        # classes
        'piedroit': 'abutment',
        'piedroit-e': 'abutment-e',
        'piedroit-w': 'abutment-w',
        'piedroit-ne': 'abutment-ne',
        'piedroit-nw': 'abutment-nw',
        'piedroit-se': 'abutment-se',
        'piedroit-sw': 'abutment-sw',
        'mur': 'wing_wall',
        'mur-se': 'wing_wall-se',
        'mur-ne': 'wing_wall-ne',
        'mur-sw': 'wing_wall-sw',
        'mur-nw': 'wing_wall-nw',
        'traverse': 'deck',
        'corniche': 'edge_beam',
        'corniche-n': 'edge_beam-n',
        'corniche-s': 'edge_beam-s',
        'gousset': 'haunch',
        'gousset-e': 'haunch-e',
        'gousset-w': 'haunch-w'
        }


@forall
def translate_keys(dictio: dict,
                w2w: dict=word2word) -> dict:
    """
    translates the keys each dictionary in a list based on the word to word translation
    of word_trans
    """
    d = flatdict.FlatDict(dictio, delimiter='.')
    for word in w2w.keys():
        pattern = f'(^|\.){word}(\.|$)'
        for key in d.keys():
            new_key = re.sub(pattern, rf'\1{w2w[word]}\2', key)
            d[new_key] = d.pop(key)

    return d.as_dict()


def translate_string(string: str,
                    w2w: dict=word2word,
                    delimiter='"') -> str:
    """
    translates the words of a string based on a dictionary. Words are enclosed by the
    delimiter.
    NOTE: delimiter should NOT a special regular expressions character, otherwise the
            behavior will be unexpected
    """
    for word in w2w.keys():
        pattern = f'({delimiter}){word}({delimiter})'
        string = re.sub(pattern, rf'\1{w2w[word]}\2', string)
    return string


@verify_format('.json')
def translate_json(json_file: str,
                    w2w: dict=word2word,
                    case_sensitive: bool=False):
    """
    loads and translates a json file based on a dictionary. All complete strings that are found
    in the file keys or values, that are part of the word2word dictionary are replaced
    by their equivalent. If case sensitive is set, capitalization matters.
    """
    with open(json_file) as f:
        string = json.dumps(json.load(f))
    if case_sensitive:
        string = translate_string(string, w2w=w2w)
    else:
        string = translate_string(string.lower(), w2w=w2w)
    return json.loads(string)

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
