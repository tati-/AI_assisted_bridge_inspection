"""
This script tests a model
example usage: python test_pipeline.py -mp /path/to/tfmodel -dp /path/to/data -cl /path/to/txtfile
"""
import os
import sys
import pdb
import argparse
import numpy as np
from natsort import natsorted

sys.path.append('..')
import modules.utils as utils
import modules.ml_utils as ml
import modules.dataset_utils as dts
from modules.decorators import timer
from modules.constants import LABELS_PIPO
exec(open('../modules/ml_imports.py').read())


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-dp', help='data path', type=str, required=True)
    arg('-mp', help='model path [savedModel tf format assumed]', type=str, nargs='+', required=True)
    arg('-bs', help='batch size [default: 8]', type=int, default=8)
    # arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)

    args = parser.parse_args()
    return args


@timer
def get_test_data(datapath: str, batch_size: int=8, labels=None) -> tuple:
    """
    given a directory containing all the dataset images, along with
    textfile containing the split info, this function returns the test set
    INPUTS:
    @datapath: a directory containing an 'images' and a 'masks' folder
    @splitpath: a text file containing 2 columns, the first including the image id
                (image name without extension) and the second containing a description
                of which set this image belongs (typically train, test, or validation)
    @batch_size: integer, batch size for generators
    @labels: a list of the label description
    @infopath: optional text file to write information in
    OUTPUTS:
    data: (x,y) tuple, containing the data and labels
    labels: a list of strings, description of classes
    """
    all_img_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                            datapath=os.path.join(datapath, 'images'))
    all_mask_paths = utils.files_with_extensions('png', 'jpg', 'JPG',
                                            datapath=os.path.join(datapath, 'masks'),
                                            recursive=True)

    if f'{os.sep}data_set3' in datapath:
        width, height = 1024, 640 #640, 480 # 1280, 640
    else:
        width, height = 640, 480
    # infer labels from directory names
    if labels is None:
        labels = list(set([os.path.normpath(x).split(os.sep)[-2] for x in all_mask_paths]))
        labels = natsorted(labels)
    # else:
    #     # inverse key-value role, so that the numbers are keys
    #     labels_dict = {val: key for key, val in utils.txt2dict(classpath).items()}
    #     # make sure the labels description matches the dictionary
    #     labels = [labels_dict[i] for i in natsorted(labels_dict.keys())]

    if 'background' not in labels:
        labels = ['background'] + labels
    # get dataframe with filepaths, one sample per row
    df = dts.organize_sample_paths(all_img_paths, all_mask_paths, labels[1:])

    img_paths = list(df.image)
    mask_paths = {key: list(df[key]) for key in df.keys()[1:]}

    data = dts.data_loader(img_paths, mask_paths, labels=labels, width=width, height=height)

    return data, labels


if __name__ == "__main__":
    ml.select_GPU()
    args = get_args()

    ############################################################################
    #                          LOAD DATA                                       #
    ############################################################################
    data, labels = get_test_data(datapath=args.dp, batch_size=args.bs, labels=LABELS_PIPO)

    ############################################################################
    #                          TEST MODEL                                     #
    ############################################################################
    ml_metrics = ml.get_segmentation_metrics()
    for model_path in args.mp:
        result_path, model_id = os.path.split(model_path.replace(f'{os.sep}models{os.sep}', f'{os.sep}results{os.sep}'))
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(metrics=ml_metrics +
                    [tf.keras.metrics.MeanIoU(num_classes=model.output.shape[-1])])
        ml.test(model, data, savedir=result_path,
                labels=labels, batch_size=args.bs, model_id=model_id, data_path=args.dp)
        del model
    """
    ############################################################################
                                    END
    ############################################################################
    """
