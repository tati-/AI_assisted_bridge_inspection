"""
############################################################################
This script initializes and trains a semantic segmentation model with the
data given as arguments. It then proceeds at testing the model at a test set
and saving the results.
example usage: python semantic_segm_pipeline.py @synthetic_config.txt
############################################################################
"""
import os
import re
import sys
import pdb
import math
import shutil
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted

sys.path.append('..')
import modules.utils as utils
import modules.ml_utils as ml
from split_dataset import split
import modules.dataset_utils as dts
import modules.visualization as vis
from modules.decorators import timer
from modules.semantic_categories import labels_pipo
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
    arg('-dtprc', help='percentage of dataset to be used for training in [0,1]', type=restricted_float, default=1)
    arg('-opt', help='Optimizer: \'SGD\',\'Adagrad\', \'Adadelta\', \'Adam\' [default: Adam]', default='Adam')
    arg('-loss', help='Loss function: mse: Mean square error,' \
                        'cce: categorical cross entropy, '\
                        'dice: dice coefficient '\
                        'cosine: cosine similarity [default: cce]', default='cce')
    arg('-ne', help='Number of epochs [default: 100]', type=int, default=100)
    arg('-bs', help='batch size [default: 8]', type=int, default=8)
    # arg('-cl', '-classes', help='path to .txt file containing the class descriptions', type=str)
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


def train_paths(datapath: str, synthetic: bool=False) -> dict:
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
    paths['info'] = os.path.join(paths['model'], 'info.txt')

    return paths


def finetune_paths(datapath: str, modelpath: str, trainable: str='all', synthetic: bool=True) -> dict:
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
    paths['info'] = os.path.join(paths['model'], 'finetune_info.txt')
    # Create directory to save results of finetuned model
    paths['results'] = paths['model'].replace(f'{os.sep}models{os.sep}',
                                                f'{os.sep}results{os.sep}')
    if os.path.isdir(paths['results']):
        shutil.rmtree(paths['results'], ignore_errors=True)
    os.makedirs(paths['results'], exist_ok=True)

    return paths


@timer
def prepare_data_for_training(datapath: str, batch_size: int=8, labels=None,
                                infopath: str=None, dataset_prc: float=1) -> tuple:
    """
    given a directory containing all the dataset images, along with
    textfile containing the split info, this function returns 3 distint sets of
    data (train, test and validation)
    INPUTS:
    @datapath: a directory containing a 'train', 'validation'(optional), and 'test' folder,
                each with an 'images' and a 'masks' folder
    @batch_size: integer, batch size for generators
    @labels: list with label descriptions
    @infopath: optional text file to write information in
    @dataset_prc: percentage of dataset to be used for training. Allows for experimenting
                    with different dataset sizes
    OUTPUTS:
    data: dictionary with keys ['train', 'validation', 'test'], each containing
            either a (x,y) tuple, or a generator producing (x,y) tuples
    labels: a list of strings, description of classes
    set_size: dictionary with keys ['train', 'validation', 'test'], containing
                the number of samples per set
    """
    sets = [x for x in ['train', 'validation', 'test'] for folder in os.scandir(datapath) if folder.name==x]
    all_files = utils.files_with_extensions('png', 'jpg', 'JPG',
                                            datapath=datapath, recursive=True)
    all_img_paths = [x for x in all_files if f'{os.sep}images{os.sep}' in x]
    all_mask_paths = [x for x in all_files if f'{os.sep}masks{os.sep}' in x]
    # for florian
    if f'{os.sep}data_set3' in datapath:
        width, height = 1024, 640 #640, 480 # 1280, 640
    else:
        width, height = 640, 480
    # width, height =  dts.min_image_size(all_img_paths)

    # infer labels from directory names
    if labels is None:
        labels = list(set(Path(x).parent.stem for x in all_mask_paths))
        labels = natsorted(labels)
    # else:
    #     # inverse key-value role, so that the numbers are keys
    #     labels_dict = {val: key for key, val in utils.txt2dict(classpath).items()}
    #     # make sure the labels description matches the dictionary
    #     labels = [labels_dict[i] for i in natsorted(labels_dict.keys())]

    if 'background' not in labels:
        labels = ['background'] + labels

    # get dataframe with filepaths, one sample per row
    csv_path = os.path.join(datapath, 'datapaths.csv')
    df = dts.organize_sample_paths(all_img_paths, all_mask_paths, labels[1:], savefile=csv_path)
    # seperate filepaths per set
    data = dict.fromkeys(sets)
    set_size = dict.fromkeys(sets)
    # load data per set
    for set_ in sets:
        set_ids = [Path(x).stem for x in utils.files_with_extensions('png', 'jpg', 'JPG',
                                                datapath=Path(datapath).joinpath(set_, 'images'))]
        if set_ in ['train', 'validation']:
            # train with smaller dataset, but maintain entire test set to
            # be able to compare
            ind = min(len(set_ids), math.ceil(len(set_ids)*dataset_prc))
            set_ids = set_ids[:ind]
        rows = pd.concat([df.loc[df.image.str.contains(f'\{os.sep}(image_)?{id}\.\w{{2,5}}$')] for id in set_ids])
        img_paths = list(rows.image)
        mask_paths = {key: list(rows[key]) for key in df.columns[1:]}
        set_size[set_] = len(img_paths)
        if set_ in ['train', 'validation']:
            # with tf 2.4, despite what the documentation says,
            # the validation data apparently can be a generator,
            data[set_] = dts.data_generator(img_paths, mask_paths,
                                                    labels=labels[1:],
                                                    width=width, height=height,
                                                    bs=batch_size)
        else:
            data[set_] = dts.data_loader(img_paths, mask_paths,
                                                    labels=labels[1:],
                                                    width=width, height=height)
        # add some info to the text file
        if infopath is None:
            continue
        with open(infopath, 'a') as f:
            f.write(f'#{set_} images: {set_size[set_]} \n\n')

    #### DEBUG: inspect dataset to make sure the data are correctly loaded #####
    # inspect_path = f'dataset_overview_test'
    # os.makedirs(inspect_path, exist_ok=True)
    # vis.inspect_dataset(data=data['test'][0], targets=np.argmax(data['test'][1], axis=-1), savefolder=inspect_path, class_desc=labels)
    # print('Dataset overview saved in {}'.format(inspect_path))
    # pdb.set_trace()
    ############################################################################

    return data, labels, set_size


def get_model(finetune: bool=False, mp: str=None, trainable: str='all', **kwargs) -> tf.keras.Model:
    """
    this function returns the tensorflow model instance to be trained
    INPUTS:
    finetune: boolean, if set implies finetuning. Mp should be not none
    mp: string, model path of model to load. Only relevant to finetuning,
        should be a pretrained model
    trainable: number of layers to train. Only relevant if finetuning
    """
    if finetune and mp is not None:
        suffix = f'{trainable}_layers'
        # load model
        model = ml.load_model_extended(mp, **kwargs)
        # model = tf.keras.models.load_model(mp, compile=False)
        # change model learning rate
        # actually for Adam the lr is already smaller, so futher
        # changing it makes it very small
        # K.set_value(model.optimizer.lr, K.get_value(model.optimizer.lr)*0.1)
        if trainable!='all' and trainable<=len(model.layers):
            for layer in model.layers[:len(model.layers)-trainable]:
                layer.trainable=False
        # it is important to recompile, otherwise the updates on the
        # trainable layers will not be incorporated
        metrics = [x for x in model.metrics if x.name!='loss']
        model._name += f'_finetuned_{suffix}'
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
    else:
        model = ml.get_network(**kwargs)

    return model


def train_model(model: tf.keras.Model, data: dict, set_size: dict, bs: int, ne: int=100,
                model_dir: str='.', results_dir: str='.', info_path: str='./info.txt',
                 **kwargs) -> tf.keras.Model:
    """
    this function initializes a model, trains it, and saves it some
    info about it
    INPUTS:
    @data: a dictionary with keys 'train', 'validation', 'test'
            train set is given in the form of a generator,
            the rest are tuples of numpy arrays in the form of (images, masks)
    @set_size: a dictionary with keys 'train', 'validation', 'test' containing
                the number of samples per set
    @bs: int, batch size
    @ne: int, number of epochs
    """
    # get info about model size
    ml.model_stats(model, info_path)

    # train model
    hist = ml.train(model=model,
                    data_train = data['train'],
                    data_val = data['validation'],
                    nBatches_per_epoch=math.ceil(set_size['train'] / bs),
                    nBatches_val = math.ceil(set_size['validation'] / bs), # if validation data generator
                    # bs=bs,  # if validation data are tuple of numpy arrays
                    nEpochs=ne,
                    savedir=model_dir)

    # plot training history
    vis.plot_training_history(hist.history, os.path.join(results_dir, 'loss.png'))

    return model


if __name__ == "__main__":
    args = get_args()
    # check if data are split for training
    subfolders =  [folder.name for folder  in os.scandir(args.dp) if folder.is_dir()]
    if len(set(subfolders).intersection(['train', 'test', 'validation']))<2:
        # no folder with data split exist, so we create a split
        warnings.warn('\n##-----## \nWarning: No split file was found so a new '\
                        'split will be created. Folders named "images" and '\
                        ' "masks" are expected.\n##-----##')
        split(args.dp, synthetic=args.synthetic)

    # the pretrained weights are for the moment only relevant for the
    # segmentation_models package
    args.pretrained = False if args.backbone is None else args.pretrained
    args.trainable = 'all' if args.trainable is None else args.trainable

    #create the folders to save models and results
    if args.finetune and args.mp is not None:
        paths = finetune_paths(args.dp, args.mp, args.trainable)
    else:
        paths = train_paths(args.dp, args.synthetic)

    info = {'data': args.dp,
            'network_architecure': args.arch,
            'backbone': args.backbone,
            'pretrained_imagenet': args.pretrained,
            'loss': args.loss,
            'optimizer': args.opt,
            'batch_size': args.bs,
            'nEpochs': args.ne,
            'classes': labels_pipo
            }
    # save experiment parameters in textfile
    utils.dict2txt(info, paths['info'])

    ############################################################################
    #                          LOAD DATA                                       #
    ############################################################################
    # get all filepaths
    data, labels, set_size = prepare_data_for_training(datapath=args.dp, batch_size=args.bs,
                                    infopath=paths['info'], labels=labels_pipo,
                                    dataset_prc=args.dtprc)
    # save labels in a text file
    labels_dict = {i:label for i,label in enumerate(labels)}
    utils.dict2txt(labels_dict, os.path.join(paths['model'], 'classes.txt'))

    ############################################################################
    #                          TRAIN MODEL                                     #
    ############################################################################
    kwargs = vars(args)
    kwargs['nClasses'] = len(labels)
    kwargs['input_shape'] = data['test'][0][0].shape
    kwargs['info_path'] = paths['info']
    kwargs['model_dir'] = paths['model']
    kwargs['results_dir'] = paths['results']
    kwargs['set_size'] = set_size
    if re.search(f'_w$', kwargs['loss']):
        # for florian
        if f'{os.sep}data_set3' in kwargs['dp']:
            kwargs['class_weights'] = np.ones(25, dtype=np.float32)
            kwargs['class_weights'][0] = 0
        else:
            kwargs['class_weights'] = [ml.class_weights[key] for key in labels]

    # initialize model
    ml.select_GPU()
    model = get_model(**kwargs)

    # train model
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    try:
        model = train_model(model, data, **kwargs)
    except Exception as e:
        shutil.rmtree(paths['model'])
        shutil.rmtree(paths['results'])
        sys.exit(f'Training failed!: {repr(e)}')

    # test final model
    ml.test(model, data['test'], savedir=paths['results'], model_id=f'{model.name}_final',
            labels=labels, batch_size=args.bs, data_path=args.dp)
    # test best validation performance model
    model = ml.load_model_extended(os.path.join(paths['model'], f'{model.name}_best_val_perf'), **kwargs)
    ml.test(model, data['test'], savedir=paths['results'], model_id=f'{model.name}_best_val_perf',
            labels=labels, batch_size=args.bs, data_path=args.dp)

    print(f'Experiment is finished, results in:\n {paths["results"]}')
    """
    ############################################################################
                                    END
    ############################################################################
    """
