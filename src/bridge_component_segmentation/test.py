"""
This script tests one or more models on a dataset
example usage: python test.py -mp /path/to/tfmodel0 /path/to/tfmodel1 -dp /path/to/data
"""
import os
import sys
import argparse
import numpy as np

sys.path.append('..')
import modules.ml_utils as ml
from modules.datasets import PFBridgeDataset
from modules.constants import LABELS_PIPO, WIDTH, HEIGHT
exec(open('../modules/ml_imports.py').read())


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-dp', help='data path', type=str, required=True)
    arg('-mp', help='model path [savedModel tf format assumed]', type=str, nargs='+', required=True)
    arg('-bs', help='batch size [default: 8]', type=int, default=8)

    args = parser.parse_args()
    return args


def test(data_path: str,
        model_paths: list,
        batch_size: int):
    """
    test a set of models on a given set of data
    """
    ml.select_GPU()

    ############################################################################
    #                          LOAD DATA                                       #
    ############################################################################
    data = PFBridgeDataset(root_path=data_path,
                                shuffle=False,
                                batch_size=batch_size,
                                img_size=(WIDTH, HEIGHT))

    ############################################################################
    #                          TEST MODEL                                     #
    ############################################################################
    ml_metrics = ml.get_segmentation_metrics()
    for model_path in model_paths:
        result_path, model_id = os.path.split(model_path.replace(f'{os.sep}models{os.sep}', f'{os.sep}results{os.sep}'))
        model = tf.keras.models.load_model(model_path, compile=False)
        model.compile(metrics=ml_metrics +
                    [tf.keras.metrics.MeanIoU(num_classes=model.output.shape[-1])])
        ml.test_model(model,
                data,
                save_path=result_path,
                model_id=model_id)
        del model


if __name__ == "__main__":
    args = get_args()
    test(data_path=args.dp,
        model_paths=args.mp,
        batch_size=args.bs)
    """
    ############################################################################
                                    END
    ############################################################################
    """
