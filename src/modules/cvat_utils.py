# some helper functions regarding manipulation of annotations coming
# from CVAT (https://cvat.org)
import os
import sys
import pdb
import cv2
import shutil
import unidecode
import numpy as np
from lxml import etree


def parse_anno_file(cvat_xml: str, image_name: str):
    """
    this function reads the annotation information for an image and returns
    it in the form of a dictionary
    """
    root = etree.parse(cvat_xml).getroot()
    anno = []
    image_name_attr = f".//image[@name='{image_name}']"
    for image_tag in root.iterfind(image_name_attr):
        image = {}
        for key, value in image_tag.items():
            image[key] = value
        image['shapes'] = []
        for poly_tag in image_tag.iter('polygon'):
            polygon = {'type': 'polygon'}
            for key, value in poly_tag.items():
                polygon[key] = value
            # for subcat in poly_tag.iter('attribute'):
            for subcat in poly_tag:
                if subcat.attrib['name'] == 'Type':
                    polygon['subcat'] = subcat.text
                    break
            image['shapes'].append(polygon)
        for box_tag in image_tag.iter('box'):
            box = {'type': 'box'}
            for key, value in box_tag.items():
                box[key] = value
            box['points'] = "{0},{1};{2},{1};{2},{3};{0},{3}".format(
                box['xtl'], box['ytl'], box['xbr'], box['ybr'])
            image['shapes'].append(box)
        image['shapes'].sort(key=lambda x: int(x.get('z_order', 0)))
        anno.append(image)
    return anno


def get_all_labels(cvat_xml: str) -> list:
    """
    this function, given an cvat xml file with annotations infers all possible
    labels (we assume that the useful labels are indicated as a attribute
    named "Type")
    """
    root = etree.parse(cvat_xml).getroot()
    labels = []
    for element in root.iter('image'):
        for polygon in element.iter('polygon'):
            for attribute in polygon:
                if attribute.attrib['name'] == 'Type':
                    labels.append(unidecode.unidecode(attribute.text).lower())
                    break
    # labels = set([x.text for el in root.iter('image') for pol in el.iter('polygon')
    #                     for x in pol if x.attrib['name'] == 'Type'])
    return list(set(labels))


def polygon2mask(width: int, height: int, polygon, scale_factor):
    """
    this functions gets a polygon(as defined from cvat annotations) and returns a binary mask
    """
    mask = np.zeros((height, width), np.uint8)
    points = [tuple(map(float, p.split(','))) for p in polygon['points'].split(';')]
    points = np.array([(int(p[0]), int(p[1])) for p in points])
    points = points*scale_factor
    points = points.astype(int)
    # mask = cv2.drawContours(mask, [points], -1, color=(255,0,0), thickness=4)
    mask = cv2.fillPoly(mask, [points], color=255)
    return mask
