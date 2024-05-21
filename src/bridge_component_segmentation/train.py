"""
############################################################################
This script initializes and trains a semantic segmentation model with the
data given as arguments.
example usage: python train.py @synthetic_config.txt
############################################################################
"""
import os
import re
import sys
import copy
import shutil
import argparse
import warnings
from pathlib import Path

sys.path.append('..')
import modules.utils as utils
import modules.ml_utils as ml
from split_dataset import split
from modules.constants import *
from modules.datasets import PFBridgeDataset
exec(open('../modules/ml_imports.py').read())


def restricted_float(x):
    """
    defines a float that is limited to be between 0 and 1
    """
    if x is None:
        return x
    try:
        x = float(x)
    except ValueError:
        raise argparse.ArgumentTypeError(f'{x} not a floating-point literal')
    if x < 0 or x > 1:
        raise argparse.ArgumentTypeError(f'{x} not in range [0, 1]')
    return x


def get_args():
    parser = argparse.ArgumentParser(fromfile_prefix_chars='@')
    arg = parser.add_argument
    arg('-dp', help='data path', type=str, required=True)
    arg('-mp', help='model path, only relevant if finetuning is set', type=str)
    arg('-opt', help='Optimizer: \'SGD\',\'Adagrad\', \'Adadelta\', \'Adam\' [default: Adam]', default='Adam')
    arg('-loss', help='Loss function: mse: Mean square error,' \
                        'cce: categorical cross entropy, '\
                        'dice: dice coefficient '\
                        'cosine: cosine similarity [default: cce]', default='cce')
    arg('-ne', help='Number of epochs [default: 100]', type=int, default=100)
    arg('-bs', help='batch size [default: 8]', type=int, default=8)
    arg('-arch', help='Network architecture to be used [default: Unet]',
                type=str, choices=['Unet', 'Unet_reduced', 'Linknet', 'FPN'], default='Unet')
    arg('-backbone', help='backbone for feature extraction of segmentation model',
                type=str, choices=ml.sm_backbones)
    arg('--pretrained', help='If set imagenet pretrained weights are used on the encoder model', action='store_true')
    arg('--synthetic', help='If set denotes that the dataset used is synthetic', action='store_true')
    arg('--finetune', help='If set denotes finetuning on an existing model', action='store_true')
    arg('-trainable', help='How many layers starting from the bottom should be trainable. Only relevant for finetuning [default:all]', type=int, default=None)

    args = parser.parse_args()
    return args


def split_dataset_in_folders(data_path: str,
                            synthetic: bool):
    """
    the dataset is expected to be divided in "train", "validation" and "test"
    folders. If it is not the case, the split is created
    INPUTS:
    @data_path: the path where all the data lies. Expected to contain either
                folders "images", "masks" or folders "train", "validation" and
                "test" each containing folders "images" and "masks"
    @synthetic: boolean describing whether the dataset is synthetic.
                This affects if the split is done by bridge or by image
    """
    subfolders =  [folder.name for folder  in os.scandir(data_path) if folder.is_dir()]
    if len(set(subfolders).intersection(['train', 'test', 'validation']))<2:
        # no folder with data split exist, so we create a split
        warnings.warn('\n##-----## \nWarning: No split file was found so a new '\
                        'split will be created. Folders named "images" and '\
                        ' "masks" are expected.\n##-----##')
        split(data_path, synthetic=synthetic)


def train_paths(datapath: str,
                synthetic: bool=False) -> dict:
    """
    this function returns the needed paths to save a model
    and its results
    INPUTS:
    @datapath: the path to dataset, used to name a similar path
    @synthetic: boolean, denoting if the training is to be done on synthetic data
    OUTPUT:
    @paths: a dictionary.
    """
    paths = dict()
    # create the folders that will contain the model and the results
    # if the data re not in a directory that contains a 'data' folder, then the
    # data_directory will serve as model and results folder - those paths could
    # also be given as arguments in the future
    model_base = datapath.replace(f'{os.sep}data{os.sep}', f'{os.sep}models{os.sep}')
    results_base = datapath.replace(f'{os.sep}data{os.sep}', f'{os.sep}results{os.sep}')
    version_id = utils.available_id(model_base, 's' if synthetic else 'r')
    paths['model'] = utils.create_folders(model_base, version_id)[0]
    paths['results'] = utils.create_folders(results_base, version_id)[0]
    paths['info'] = os.path.join(paths['model'], 'info.config')

    return paths


def finetune_paths(datapath: str,
                modelpath: str,
                trainable: str='all',
                synthetic: bool=True) -> dict:
    """
    this function returns the needed paths to save a finetuned model
    and its results
    INPUTS:
    @modelpath: the path to the pretrained model, a directory of savedModel tf
                format
    trainable: int, the number of network layers to finetune. If None or
                higher than the total number of layers, all layers are retrained
    OUTPUT:
    @paths: a dictionary.
    """
    paths = dict()
    dir_list = os.path.normpath(modelpath).split(os.sep)
    # directory where the model in SavedModel format lies
    model_base_dir = os.path.join(*dir_list[:-1])

    suffix = f'{trainable}_layers'

    # create directory for finetuned model
    paths['model'] = utils.create_folders(model_base_dir,
                    f'{dir_list[-1]}_finetuned_with_{Path(datapath).stem}_{suffix}')[0]
    # create a text file for some info
    paths['info'] = os.path.join(paths['model'], 'info.config')
    # Create directory to save results of finetuned model
    paths['results'] = paths['model'].replace(f'{os.sep}models{os.sep}',
                                                f'{os.sep}results{os.sep}')
    if os.path.isdir(paths['results']):
        shutil.rmtree(paths['results'], ignore_errors=True)
    os.makedirs(paths['results'], exist_ok=True)

    return paths


def train(data_path: str,
            model_path: str=None,
            optimizer: str='Adam',
            loss: str='cce',
            batch_size: int=8,
            n_epochs: int=50,
            architecture: str='Unet',
            backbone: str=None,
            pretrained: bool=False,
            synthetic: bool=False,
            finetune: bool=False,
            trainable: int=None,
            ):
    """
    trains a model described by the arguments on a dataset
    """
    # split dataset in folder if not done
    split_dataset_in_folders(data_path, synthetic)
    # the pretrained weights are for the moment only relevant for the
    # segmentation_models package
    pretrained = False if backbone is None else pretrained
    #create the folders to save models and results
    if finetune and model_path is not None:
        trainable = 'all' if trainable is None else trainable
        paths = finetune_paths(data_path, model_path, trainable)
    else:
        paths = train_paths(data_path, synthetic)
    info = copy.deepcopy(locals())
    del info['paths']
    # save experiment parameters in config file
    utils.dict2txt(info, paths['info'])

    ############################################################################
    #                          LOAD DATA                                       #
    ############################################################################
    data = dict()
    for set_ in ['train', 'validation']:
        kwargs = {'root_path': Path(data_path).joinpath(set_),
                    'shuffle': True,
                    'batch_size': batch_size,
                    'img_size': (WIDTH, HEIGHT)
                }
        data[set_] = PFBridgeDataset(**kwargs)

    ############################################################################
    #                          TRAIN MODEL                                     #
    ############################################################################
    kwargs = info
    kwargs['n_classes'] = len(LABELS_PIPO)
    kwargs['input_shape'] = (HEIGHT, WIDTH, 3)
    kwargs['info_path'] = paths['info']
    if re.search(f'_w$', kwargs['loss']):
        kwargs['class_weights'] = [CLASS_WEIGHTS[key] for key in LABELS_PIPO]

    # initialize model
    ml.select_GPU()
    model = ml.get_model(**kwargs)
    # model.save(Path(paths['model']).joinpath(f'{model.name}_untrained'))
    # train model
    try:
        model = ml.train_model(model=model,
                                data_train=data['train'],
                                data_val=data['validation'],
                                n_epochs=n_epochs,
                                save_path=paths['model'])
    except Exception as e:
        shutil.rmtree(paths['model'])
        shutil.rmtree(paths['results'])
        sys.exit(f'Training failed!: {repr(e)}')

    print(f'Training is finished, model in:\n {paths["model"]}')


if __name__ == "__main__":
    args = get_args()
    train(data_path=args.dp,
            model_path=args.mp,
            optimizer=args.opt,
            loss=args.loss,
            batch_size=args.bs,
            n_epochs=args.ne,
            architecture=args.arch,
            backbone=args.backbone,
            pretrained=args.pretrained,
            synthetic=args.synthetic,
            finetune=args.finetune,
            trainable=args.trainable
            )
    """
    ############################################################################
                                    END
    ############################################################################
    """
