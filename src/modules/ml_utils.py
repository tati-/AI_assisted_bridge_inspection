import os
import re
import sys
import cv2
import math
import shutil
import random
import inspect
import warnings
import numpy as np
from tqdm import tqdm
from scipy import stats
from pathlib import Path
# import numpy.typing as npt
import matplotlib.pyplot as plt
import segmentation_models as sm
from sklearn.metrics import confusion_matrix, f1_score

from . import utils
from .decorators import optional, timer
from .constants import LABELS_PIPO
from . import visualization as vis
from . import metric_utils as metrics
from . import multiclass_losses as multiloss
exec(open('../modules/ml_imports.py').read())


sm_backbones = ['vgg16', 'vgg19',
                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152',
                'resnext50', 'resnext101',
                'seresnext50', 'seresnext101',
                'senet154',
                'densenet121', 'densenet169', 'densenet201',
                'inceptionv3', 'inceptionresnetv2',
                'mobilenet', 'mobilenetv2',
                'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3',
                'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'
                ]


@optional
def select_GPU():
    gpus = tf.config.list_physical_devices('GPU')
    if len(gpus)==0:
        print('#'*10)
        print('NO GPU IS USED!!!')
        print('#'*10)
        return
        # Hide GPU from visible devices
        # tf.config.set_visible_devices([], 'GPU')
    gpu_index = 0
    max_mem = 0
    nvidia_smi.nvmlInit()
    for i in range(len(gpus)):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        if info.free>max_mem:
            max_mem = info.free
            gpu_index = i
        # print(f'Total memory:{info.total}')
        print(f'{gpus[i].name}: Free memory:{info.free}')
        # print(f'Used memory:{info.used}')
    nvidia_smi.nvmlShutdown()
    # tf.test.gpu_device_name()
    tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
    # https://www.tensorflow.org/guide/gpu


def get_segmentation_metrics() -> list:
    """
    This is a function and not a globar variable, because
    if it is a global variable the select_gpu function fails
    with RuntimeError: Physical devices cannot be modified after being initialized
    TODO: there is some issue with those metrics. they are not consistent with the
        confusion matrix, at least for tf version 2.4
    """
    segmentation_metrics = [tf.keras.metrics.CategoricalAccuracy(),
                            tf.keras.metrics.TruePositives(),
                            tf.keras.metrics.FalsePositives(),
                            tf.keras.metrics.FalseNegatives(),
                            tf.keras.metrics.Precision(),
                            tf.keras.metrics.Recall(),
                            tf.keras.metrics.CategoricalCrossentropy()
                            ]
    return segmentation_metrics


def get_loss(loss_name: str, class_weights=None):
    losses = {'cce': tf.losses.CategoricalCrossentropy(),
            'cce_w': multiloss.multiclass_weighted_cross_entropy(class_weights),
            'mse': tf.losses.MeanSquaredError(),
            'mae': tf.losses.MeanAbsoluteError(),
            'dice': multiloss.multiclass_weighted_dice_loss(),
            'dice_w': multiloss.multiclass_weighted_dice_loss(class_weights),
            'cosine': tf.losses.CosineSimilarity(),
            'tanimoto': multiloss.multiclass_weighted_tanimoto_loss(),
            'tanimoto_w': multiloss.multiclass_weighted_tanimoto_loss(class_weights)
            }

    if loss_name in losses.keys():
        return losses[loss_name]
    else:
        sys.exit(f'loss {loss_name} not defined')


def get_network(architecture: str,
                backbone: str=None,
                input_shape: tuple=(None, None, 3),
                pretrained: bool=False,
                n_classes:int=2,
                optimizer: str='Adam',
                loss: str='cce',
                class_weights=None,
                **kwargs):
    """
    This function returns a network object defined by network
    INPUTS:
    @architecture	: a string indicating what network architecture will be used
    @backbone   : a string indicating the backbone (relevant for segmentation_models
                    package)
    @input_shape: in the case that a CNN model will be trained, the input shape
    			  (it is necessary to build the model)
    @n_classes   : the number of output classes
    @class_weights: array, class weights
    OUTPUT
    @classifier : a classifier object
    """
    loss = get_loss(loss, class_weights)
    if optimizer == 'SGD':
        optimizer = SGD(lr=lera, decay=1e-6, momentum=0.9, nesterov=True)
    encoder_weights='imagenet' if pretrained else None

    if architecture == 'Unet' and backbone is None:
        # unet network, https://github.com/jocicmarko/ultrasound-nerve-segmentation
        inputs = Input(input_shape, name='input')
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        encoded = Conv2D(512, (3, 3), activation='relu', padding='same', name='encoded')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(encoded), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', name='mask')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10], name=arch)

    elif architecture == 'Unet':
        model = sm.Unet(backbone_name=backbone, input_shape=input_shape,
                        classes=n_classes, encoder_weights=encoder_weights)

    elif architecture == 'Unet_reduced':
        # unet network, https://github.com/jocicmarko/ultrasound-nerve-segmentation
        inputs = Input(input_shape, name='input')
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        encoded = Conv2D(256, (3, 3), activation='relu', padding='same', name='encoded')(conv4)

        up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoded), conv3], axis=3)
        # up5 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(encoded)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(up5)
        conv5 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv5), conv2], axis=3)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv6), conv1], axis=3)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv7)

        conv8 = Conv2D(n_classes, (1, 1), activation='sigmoid', name='reconstructed')(conv7)

        model = Model(inputs=[inputs], outputs=[conv8], name=arch)

    elif architecture=='Linknet':
        # Linknet is a fully convolution neural network for fast image semantic segmentation
        # This implementation by default has 4 skip connections (original - 3).
        model = sm.Linknet(backbone_name=backbone, input_shape=input_shape,
                        classes=n_classes, encoder_weights=encoder_weights)

    elif architecture=='FPN':
        # FPN is a fully convolution neural network for image semantic segmentation
        model = sm.FPN(backbone_name=backbone, input_shape=input_shape,
                        classes=n_classes, encoder_weights=encoder_weights)

    elif architecture=='PSPNet':
        # PSPNet is a fully convolution neural network for image semantic segmentation
        # shape cannot be none! should be divisible by 6 * downsample_factor
        if any([x is None for x in input_shape]):
            model = sm.PSPNet(backbone_name=backbone,classes=n_classes,
                            encoder_weights=encoder_weights)
        else:
            model = sm.PSPNet(backbone_name=backbone, input_shape=input_shape,
                            classes=n_classes, encoder_weights=encoder_weights)

    else:
        sys.exit(f'{architecture} type of network architecture not defined')

    name = f'{architecture}'
    if backbone is not None:
        name += f'_{backbone}'
    if pretrained:
        name += '_pr'
    model._name = name
    model.compile(optimizer=optimizer, loss=loss,
                metrics=get_segmentation_metrics() +
                [tf.keras.metrics.MeanIoU(num_classes=model.output.shape[-1])])
    return model


def load_model_extended(model_path: str,
                        loss: str=None,
                        class_weights: dict=None,
                        **kwargs):
    """
    this function loads a model taking into account that potentially a custom loss
    function is used
    """
    tf_losses = [x[0] for x in inspect.getmembers(tf.losses)]
    if loss in tf_losses:
        return tf.keras.models.load_model(model_path)

    # dummy load in order to get the number of classes
    # if class_weights is None:
    #     model = tf.keras.models.load_model(modelpath, compile=False)
    #     n_classes = model.output.shape[-1]
    #     class_weights = np.ones(n_classes).astype(np.float32)

    custom = {'loss': get_loss(loss, class_weights)}
    return tf.keras.models.load_model(model_path, custom_objects=custom)


def get_model(finetune: bool=False,
                model_path: str=None,
                trainable: str='all',
                **kwargs) -> tf.keras.Model:
    """
    this function returns the tensorflow model instance to be trained
    INPUTS:
    finetune: boolean, if set implies finetuning. Mp should be not none
    model_path: string, model path of model to load. Only relevant to finetuning,
        should be a pretrained model
    trainable: number of layers to train. Only relevant if finetuning
    **kwargs: arguments to be passed to get_network
    """
    if finetune and model_path is not None:
        suffix = f'{trainable}_layers'
        # load model
        model = load_model_extended(model_path, **kwargs)
        if trainable!='all' and trainable<=len(model.layers):
            for layer in model.layers[:len(model.layers)-trainable]:
                layer.trainable=False
        # it is important to recompile, otherwise the updates on the
        # trainable layers will not be incorporated
        metrics = [x for x in model.metrics if x.name!='loss']
        model._name += f'_finetuned_{suffix}'
        model.compile(optimizer=model.optimizer, loss=model.loss, metrics=metrics)
    else:
        model = get_network(**kwargs)

    return model


def model_stats(model: tf.keras.Model,
                info_path: str=None):
    """
    this function calculates and prints some info on the model.
    If a savefile is provided, that info is written there
    """
    model.summary()
    trainable_params = np.sum([np.prod(v.get_shape()) for v in model.trainable_weights])
    non_trainable_params = np.sum([np.prod(v.get_shape()) for v in model.non_trainable_weights])
    total_params = trainable_params + non_trainable_params
    if info_path is not None:
        with open(info_path, 'a') as f:
            f.write(f'#trainable parameters: {trainable_params} \n')
            f.write(f'#non trainable parameters: {non_trainable_params} \n')
            f.write(f'#parameters: {total_params} \n\n')
    # return trainable_params, non_trainable_params


class EvaluateCallback(tensorflow.keras.callbacks.Callback):
    """
    This callback class passes an image through the network during training, and saves the
    network output (save_every defines the number of epochs after which a new prediction is made).
    It serves as a monitor to see how a network slowly learns to identify the interesting structures.
    """
    def __init__(self,
                 original,
                 truth,
                 out_path=None,
                 save_every=3,
                 colors=None):
        tf.keras.callbacks.Callback.__init__(self)
        self.n_batch = 0
        self.n_epoch = 0
        self.out_path = out_path
        self.save_every = save_every
        # self.verbose = verbose
        # cv2 imwrite requires integers in [0, 255] range.
        self.image = np.expand_dims(original, 0)
        self.truth = np.expand_dims(truth, 0)
        if colors is None:
            # get values of preexisting colormap as list
            # self.colors = cm.get_cmap('tab20')((np.linspace(0, 1, 20)))[:, :3]
            # self.colors[0, :] = np.asarray([0,0,0])
            self.colors = vis.class_colors
        else:
            self.colors = colors
		# create path to save intermediate results, if it does not exist
        os.makedirs(out_path, exist_ok=True)

        if out_path is not None:
            plt.imsave(os.path.join(self.out_path, 'original.png'), original[...,:3])
            gt_categorical = np.argmax(np.squeeze(truth), axis=-1).astype('uint8') # for categorical groundtruth
            # cv2 imwrite requires an image of integers in the range of [0,255]
            gt_img = vis.categorical2color_img(gt_categorical, self.colors)
            plt.imsave(os.path.join(self.out_path, 'truth.png'), gt_img)

    # def on_batch_end(self, batch, logs={}):
    def on_epoch_end(self, batch, logs={}):
        if self.n_epoch % self.save_every == 0:
            if self.out_path is not None:
                output = self.model.predict(self.image, batch_size=1)
                output = np.argmax(np.squeeze(output), axis=-1).astype('uint8')
                output = vis.categorical2color_img(output, self.colors)
                plt.imsave(os.path.join(self.out_path,f'epoch_{self.n_epoch}_output.png'),
                    output)
        self.n_epoch += 1


def train_model(model: tf.keras.Model,
            data_train: tf.keras.utils.Sequence,
            data_val: tf.keras.utils.Sequence=None,
            n_epochs: int=100,
            save_path: str='.',
            fold: int=None):
    """
    trains the network. Training and validation data and groundtruths
    are needed.
    INPUTS:
    @model: a tf keras model
    @data_train: a tf.keras.utils.Sequence object producing (x,y) tuples of some batch size
    @data_val: a tf.keras.utils.Sequence object producing (x,y) tuples of some batch size
    @n_epochs: integer, number of epochs
    @save_path: directory path to save the procuded model
    @fold: fold number, if cross-validation scheme is used
    OUTPUTS:
    @hist: history object
    """
    model_stats(model, Path(save_path).joinpath('info.config'))

    # CALLBACKS
    callbck = []
    # fit arguments
    fit_args = dict()
    fit_args['x'] = data_train
    fit_args['validation_data'] = data_val
    fit_args['verbose'] = 1
    fit_args['epochs'] = n_epochs
    # save intermediate training steps (every 25 batches)
    result_save_path = save_path.replace(f'{os.sep}models{os.sep}', f'{os.sep}results{os.sep}')
    inter_path = os.path.join(result_save_path, 'intermediate_results')
    inter_sample = data_val.sample_loader(0)
    inter = EvaluateCallback(inter_sample[0],
                            inter_sample[1],
                            out_path = inter_path,
                            colors=vis.class_colors)
    callbck.append(inter)
    #
    base_folder = f'{model.name}_best_val_perf' if fold is None else f'{model.name}_best_val_perf_fold_{fold}'
    file_path = os.path.join(save_path, base_folder) # If i have cross validation I can add the fold
    model_checkpoint = ModelCheckpoint(file_path,
                                        monitor='val_loss',
                                        verbose=1,
                                        save_best_only=True)
    callbck.append(model_checkpoint)
    #
    patience = 60 if 'finetuned' in model.name else 40
    early_stopping = EarlyStopping(patience=patience,
                                restore_best_weights=False) # start_from_epoch=20
    callbck.append(early_stopping)
    #
    match = re.search(f'(^\{os.sep}?([\w.]+\{os.sep})+(results))\{os.sep}(.+)$',
                    result_save_path)
    log_dir = os.path.join(match.group(1), 'logs', match.group(4))
    if os.path.isdir(log_dir):
        shutil.rmtree(log_dir, ignore_errors=True) # clean directory if exists
    tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1, write_images=True)
    callbck.append(tensorboard)
    # callbck.append(TqdmCallback(verbose=2))
    fit_args['callbacks'] = callbck
    hist = model.fit(**fit_args)

    # also save after last epoch
    save_name = f'{model.name}_final' if fold is None else f'{model.name}_final_fold_{fold}'
    model.save(os.path.join(save_path, save_name))
    np.save(os.path.join(result_save_path, 'training_history.npy'), hist.history)

    vis.plot_training_history(hist.history,
                        Path(result_save_path).joinpath('loss.png'))

    return hist


@timer
def test_model(model: tf.keras.Model,
        data: tf.keras.utils.Sequence,
        save_path: str='.',
        model_id: str=''):
    """
    tests a model on a test set
    INPUTS
    @model: tf keras model
    @data: a tf.keras.utils.Sequence object producing (x,y) tuples of some batch size
    @model_id: string, specificator for the model (eg final, best_val, etc)
    """
    save_path = os.path.join(save_path, model_id)
    os.makedirs(save_path, exist_ok=True)
    # keep track on which dataset the model is being tested
    data_id = os.path.normpath(data.root_path).split(os.sep)[-1]
    data_id = os.path.normpath(data.root_path).split(os.sep)[-2] if data_id=='test' else data_id

    match = re.search(f'(^\{os.sep}?([\w.]+\{os.sep})+)((s|r)?v\d{{4}})', save_path)
    csv_path = os.path.join(match.group(1), 'result_overview.csv')

    test_overview = {'model': f'{match.group(3)}_{model_id}', 'tested_on': data.root_path}

    # test_results = model.evaluate(data)
        # # TODO: there is an issue with the tf metrics, at least for version tf 2.4,
        # # they are not reliable and consistent with the confusion matrix, So I prefer
        # # to calculate them by hand (I think it has to do with the way they
        # # handle the per batch results)
        # test_overview.update({name: test_results[i] for i, name in enumerate(model.metrics_names)
        #                 if name!='loss'})
    # !!!
    images = np.asarray([data.sample_loader(i)[0] for i in range(data.n_samples)])
    y_binary = np.asarray([data.sample_loader(i)[1] for i in range(data.n_samples)])
    y_cat = np.argmax(np.squeeze(y_binary), axis=-1)
    del y_binary

    # predict on test data
    # the concatenation of the batch predicted outputs happens in the GPU, which
    # causes an OOM (out of memory) error. This is why we predict on batches and
    # manually concatenate
    try:
        # predict on batches and concatenate manually
        pred = list()
        for i in tqdm(range(math.ceil(data.n_samples/data.batch_size)),
                        'Predicting on test set:'):
            start = i*data.batch_size
            end = min(i*data.batch_size+data.batch_size, data.n_samples)
            pred.extend(model.predict_on_batch(images[start:end]))
        pred=np.asarray(pred)
    except:
        # if this fails, predict on CPU
        warnings.warn('Predicting on CPU - This process is expected to be slow \n')
        with tf.device("cpu:0"):
            pred = model.predict(images, batch_size=data.batch_size)
    # get class id from categorical data
    pred_cat = np.argmax(np.squeeze(pred), axis=-1)

    #confusion matrix
    conf_matr = confusion_matrix(y_true=y_cat.flatten(),
                y_pred=pred_cat.flatten(),
                labels=list(range(len(LABELS_PIPO))))

    # test = tf.math.confusion_matrix(y_cat.flatten(), pred_cat.flatten())
    save_file = os.path.join(save_path, f'{data_id}_confusion_matrix_{model_id}.pdf')
    vis.plot_confusion_matrix(conf_matr, LABELS_PIPO, save_file)
    for norm in ['true', 'pred', 'all']:
        conf_matr = confusion_matrix(y_true=y_cat.flatten(),
                                    y_pred=pred_cat.flatten(),
                                    labels=list(range(len(LABELS_PIPO))),
                                    normalize=norm)
        save_file = os.path.join(save_path, f'{data_id}_confusion_matrix_{model_id}_normalized_{norm}.pdf')
        vis.plot_confusion_matrix(conf_matr, LABELS_PIPO, save_file)
    # METRICS
    f1_per_class = metrics.fscore_from_cm(conf_matr)
    for i, label in enumerate(LABELS_PIPO):
        test_overview[f'{label}_F1score'] = f1_per_class[i]
    test_overview['precision_macro'] =metrics.precision_from_cm(conf_matr, mean=True)
    test_overview['recall_macro'] = metrics.recall_from_cm(conf_matr, mean=True)
    test_overview['F1score_macro'] = np.mean(f1_per_class)
    test_overview['mean_IoU'] = metrics.IoU_from_cm(conf_matr, mean=True)

    print(f'{"#"*10}{model_id}{"#"*10}')
    for metric, value in  test_overview.items():
        print(f'{metric} : {value}')
    print(f'{"#"*30}')

    # save result summary in csv file
    utils.dict2csv(test_overview, csv_path)
    # save test images with groundtruth and prediction for inspections
    test_images_path = os.path.join(save_path, f'{data_id}_test_images_results_{model_id}')
    os.makedirs(test_images_path, exist_ok=True)
    vis.inspect_predictions(*[(images[i], y_cat[i], pred_cat[i]) for i in range(data.n_samples)],
                            labels=LABELS_PIPO,
                            savefolder=test_images_path)
