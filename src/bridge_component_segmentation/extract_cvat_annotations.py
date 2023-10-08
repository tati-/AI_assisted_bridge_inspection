"""
This script given an image set and an xml file exported from CVAT annotation
tool (https://cvat.org/), CVAT for images format (one xml file containing all
annotations), loads the annotations and saves them in the form of binary masks.
It then creates some overview images with each image, its annotations, and the
label description.

inspired by https://towardsdatascience.com/extract-annotations-from-cvat-xml-file-into-mask-files-in-python-bb69749c4dc9
example usage local : python extract_cvat_annotations.py --image-dir path/to/image/directory --cvat-xml path/to/xml
"""
import os
import sys
import pdb
import cv2
import shutil
import warnings
import argparse
import unidecode # unidecode.unidecode(string)
import numpy as np
from tqdm import tqdm
from pathlib import Path
from natsort import natsorted

sys.path.append('..')
import modules.utils as utils
import modules.cvat_utils as cvat
import modules.dataset_utils as dts
import modules.visualization as vis
from modules.semantic_categories import labels_pipo

def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('--image-dir', metavar='DIRECTORY',
        default='../../data/cvat_dataset/images',
        help='directory with input images [default: ../../data/cvat_dataset/images]')
    arg('--cvat-xml', metavar='FILE',
        default='../../data/cvat_dataset/annotations.xml',
        help='input file with CVAT annotation in xml format [default: ../../data/cvat_dataset/annotations.xml]')
    arg('--scale-factor', type=float, default=1.0,
        help='choose scale factor for images [default: 1]')
    # arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)


    args = parser.parse_args()
    return args


def xml2masks(xml: str, img_dir: str, save_dir: str, labels: list=None, scale_factor: float=1.0) -> list:
    """
    this function, given an xml file with all the cvat labels, creates a folder
    per label in savedir and saves binary masks
    INPUTS:
    @xml: path to cvat xml file
    @img_dir: directory where the images are saved
    @save_dir: directory to save the binary masks
    @labels: a list of labels of interest. If None, all labels appearing in the
            xml are used
    OUTPUTS:
    a list of labels
    """
    all_labels = cvat.get_all_labels(xml)
    labels = all_labels if labels is None else labels
    img_paths = utils.files_with_extensions('png', 'JPG', 'jpg', datapath=img_dir)

    # clean up directory for masks if it already exists
    # the ignore_errors flag is set to True because of a strange behavior when
    # the script is run from mac, check
    # https://stackoverflow.com/questions/64687047/shutil-rmtree-filenotfounderror-errno-2-no-such-file-or-directory-xxx
    # if os.path.isdir(save_dir): shutil.rmtree(save_dir, ignore_errors=True)
    os.makedirs(save_dir, exist_ok=True)

    for img_path in tqdm(img_paths, desc='Reading annotations:'):
        # get annotations for a single image (originally should have a single element,
        # but it may occur that a single image has more than one annotation entries in
        # the xml file)
        info_1image = cvat.parse_anno_file(xml, Path(img_path).name)
        if not info_1image:
            os.remove(img_path)
            continue
        for img_occurence, annotation in enumerate(info_1image):
            if img_occurence==0:
                # current_image = cv2.imread(img_path)
                height, width = int(annotation['height']), int(annotation['width'])
                masks = np.zeros((height, width, len(labels)), np.uint8)
            for shape in annotation['shapes']:
                if shape['subcat'] == 'Autre':
                    # if it goes in here go back and annotate the indicated file
                    print(f'Annotation for {annotation["name"]} is not well defined')
                    pdb.set_trace()
                    continue
                elif unidecode.unidecode(shape['subcat']).lower() not in all_labels:
                    sys.exit(f'{shape["subcat"]} not recognised as a label')
                ind = [i for i,lab in enumerate(labels)
                        if lab in unidecode.unidecode(shape['subcat']).lower()]
                if len(ind)==1:
                    masks[:,:,ind] += cvat.polygon2mask(width, height,shape, scale_factor)[..., None]
                else:
                    warnings.warn(f'\n{shape["subcat"]} corresponded to {len(ind)} labels, so it was ignored\n')
        # if no mask at all is found, delete image and move to the next image
        if np.sum(masks) == 0:
            os.remove(img_path)
            continue
        for i in range(masks.shape[-1]):
            # savefolder = os.path.join(output_dir, unidecode.unidecode(labels[i].lower()))
            savefolder = os.path.join(save_dir, labels[i])
            os.makedirs(savefolder, exist_ok=True)
            savepath = os.path.join(savefolder, f'mask_{Path(img_path).stem}.png')
            cv2.imwrite(savepath, masks[...,i])

    return labels


if __name__ == "__main__":
    args = get_args()
    assert os.path.splitext(args.image_dir)[1]=='', '--image-dir should be a directory'
    assert os.path.splitext(args.cvat_xml)[1]=='.xml', '--cvat-xml should be an xml file'
    # if args.cl is not None:
    #     # inverse key-value role, so that the numbers are keys
    #     labels_dict = {val: key for key, val in utils.txt2dict(args.cl).items()}
    #     # make sure the labels description matches the dictionary
    #     labels = [labels_dict[i] for i in natsorted(labels_dict.keys())]
    # else:
    #     labels = None
    # define directory to put masks in
    base_dir = Path(args.image_dir).parent
    output_dir = os.path.join(base_dir, 'masks')

    # infer masks from polygons of xml file and save them
    labels = xml2masks(xml=args.cvat_xml, img_dir=args.image_dir,
                        save_dir=output_dir, labels=labels_pipo, scale_factor=args.scale_factor)

    ############################################################################
    #                         LOAD AND SHOW IMAGES                             #
    ############################################################################
    image_paths = utils.files_with_extensions('jpg', 'JPG', 'png', datapath=args.image_dir)
    image_paths = [p for p in image_paths if not os.path.basename(p).startswith('._')]
    mask_paths =  utils.files_with_extensions('png', datapath=os.path.join(base_dir, 'masks'),
                                                recursive=True)
    if 'background' not in labels:
        labels = ['background'] + labels
    df = dts.organize_sample_paths(image_paths, mask_paths)

    overview_dir = os.path.join(base_dir, 'overview')
    os.makedirs(overview_dir, exist_ok=True)

    vis.inspect_dataset(*[row for i,row in df.iterrows()], labels=labels, savefolder=overview_dir)
"""
############################################################################
                                END
############################################################################
"""
