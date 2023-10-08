import os
import sys
import pdb
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
sns.set(font='DejaVu Serif', font_scale=2)
sns.set_style('darkgrid', {'font.family':'serif'})

from . import utils
from . import dataset_utils as dts
from .decorators import forall


# get values of preexisting colormap as list
class_colors = cm.get_cmap('tab20')((np.linspace(0, 1, 12)))[:, :3]
class_colors[0, :] = np.asarray([0,0,0])


def categorical2color_img(cat_img, colors):
    """
    this function takes a categorical 2d map, with integers representing different
    categories, and a list of colors (3d).
    It then creates a colored 3d image, where each category is represented by a
    different color of the list. The categories basically serve as indices to the
    color list.
    """
    colored_img = np.zeros((cat_img.shape + (colors.shape[-1],)))
    for i, color in enumerate(colors):
        colored_img[cat_img==i, ...] = colors[i, ...]

    return colored_img


def plot_time_wrt_vertices(time_, n_vertices, save_path='time_wrt_vertices.pdf', time_unit='sec'):
    """
    plots one or more (depending on input) curves
    """
    # plot and save render time with respect to number of vertices of mesh
    if isinstance(time_, dict):
        gs = GridSpec(nrows=len(time_.keys()), ncols=1)
        fig = plt.figure(figsize=(6.5, 6.5*len(time_.keys())))
        for i, (key, value) in enumerate(time_.items()):
            ax = fig.add_subplot(gs[i, 0])
            ax.plot(n_vertices, value, 'o-', mfc='none')
            ax.set_title('{} time with respect to number of vertices'.format(key))
            ax.set_xlabel('Number of vertices')
            ax.set_ylabel('Time [{}]'.format(time_unit))
    else:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(n_vertices, time_, 'o-', mfc='none')
        ax.set_title('Render time with respect to number of vertices')
        ax.set_xlabel('Number of vertices')
        ax.set_ylabel('Time [{}]'.format(time_unit))
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


@forall
def inspect_dataset(paths: pd.Series, labels, savefolder=None):
    """
    this function displays each data sample image with its groundtruth
    If a savefolder is given as input, those images are saved.
    INPUTS:
    @paths: pandas series, sample paths with keys [image, label1, label2, ...]
                    for a single sample
    @labels: list of strings, labels description. The index of each list element
            corresponds to the integer value representing the label. Background
            label should be included
    @savefolder: path to folder where overview should be saved
    """
    # create colormap with specific number of bins
    cmap = LinearSegmentedColormap.from_list('cm', class_colors, N=len(labels))
    mask_paths = {key: [paths[key]] for key in paths.keys()[1:]}

    im, gt = dts.data_loader([paths.image], mask_paths, labels=labels)
    # get number of pixel labels per class
    class_concentr = np.sum(gt, axis=tuple(range(gt.ndim-1)))
    class_concentr = {label: int(class_concentr[i]) for i,label in enumerate(labels)}
    #
    # transform mask array from one hot encoding to ''discrete'' encoding
    # [[0, 1, 0], [1, 0, 0], [0, 0, 1]] -> [1, 0, 2]
    im, gt = im[0], np.argmax(gt[0], axis=-1)
    # convert image to int if in [0-255] range
    im = im.astype(int) if np.max(im)>1 else im
    im_id = Path(paths.image).stem
    # im_id = os.path.splitext(os.path.basename(paths.image))[0].replace('image_', '')
    # if im_id=='':
    #     pdb.set_trace()
    fig = plt.figure(figsize=(3*len(labels),8)) #(15,8)
    gs = GridSpec(nrows=2, ncols=2, height_ratios=[1, 0.1])
    # ax = np.empty(3, dtype=object)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_gt = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[1, :])
    ax_rgb.imshow(im)
    ax_rgb.set_title('Original image')
    ax_rgb.grid(False)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    #
    im_gt = ax_gt.imshow(gt, vmin=0, vmax=len(labels)-1, cmap=cmap) #, cmap=cmap
    ax_gt.set_title('Groundtruth segmentation')
    ax_gt.grid(False)
    ax_gt.set_xticks([])
    ax_gt.set_yticks([])
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    # divider = make_axes_locatable(ax_gt)
    # cax = divider.append_axes("right", size="5%", pad=0.05)
    # plt.colorbar(im_gt, cax=cax, ticks=range(len(gt_labels)))
    # cax.set_yticklabels(gt_labels)  # vertical colorbar
    ax_cb.grid(False)
    bin_size = (len(labels)-1)/len(labels)
    ticks = [x*bin_size+bin_size/2 for x in range(len(labels))]
    fig.colorbar(im_gt, cax=ax_cb, orientation='horizontal', ticks=ticks)
    ax_cb.set_xticklabels(labels, fontsize='x-small') # horizontal colorbar
    if savefolder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savefolder, f'{im_id}.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savefolder, f'{im_id}.pdf'), bbox_inches='tight')
    plt.close()

    return class_concentr


@forall
def inspect_predictions(data: tuple, labels=None, savefolder=None):
    """
    this function displays a data sample image with its groundtruth(optional) and
    with the network prediction. If a savefolder is given as input, those images are saved.
    INPUTS:
    @data: tuple containing 3 numpy arrays (image[3d], target[2d or None], prediction[2d])
    @labels: a list of label description
    @savefolder: string, directory where the produced figure is to be saved
    """
    img, target, prediction = data
    assert img is not None and prediction is not None, 'only the groundtruth mask can be ommited!!'
    # see how many subplots there will be
    nSubplots = np.sum([x is not None for x in [img, target, prediction]])
    labels = list(range(np.max(target))) if labels is None else labels
    cmap = LinearSegmentedColormap.from_list('cm', class_colors, N=len(labels))
    fig = plt.figure(figsize=(3*len(labels)+7*(nSubplots-2),8))
    gs = GridSpec(nrows=2, ncols=nSubplots, height_ratios=[1, 0.1])
    # ax = np.empty(3, dtype=object)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(img)
    ax_rgb.set_title('Original image')
    #
    if target is not None:
        ax_gt = fig.add_subplot(gs[0, 1])
        ax_gt.imshow(target, vmin=0, vmax=len(labels)-1, cmap=cmap)
        ax_gt.set_title('Groundtruth segmentation')
    #
    ax_pred = fig.add_subplot(gs[0, nSubplots-1])
    im_cb = ax_pred.imshow(prediction, vmin=0, vmax=len(labels)-1, cmap=cmap)
    ax_pred.set_title('Prediction')
    #
    for ax in fig.get_axes():
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    ax_cb = fig.add_subplot(gs[1, :])
    ax_cb.grid(False)
    bin_size = (len(labels)-1)/len(labels)
    ticks = [x*bin_size+bin_size/2 for x in range(len(labels))]
    fig.colorbar(im_cb, cax=ax_cb, orientation='horizontal', ticks=ticks)
    ax_cb.set_xticklabels(labels, fontsize='x-small') # horizontal colorbar
    if savefolder is None:
        plt.show()
    else:
        id = utils.last_file_index(glob.glob(os.path.join(savefolder, '*.png'))) + 1
        plt.savefig(os.path.join(savefolder, f'{id}.pdf'), bbox_inches='tight')
        plt.savefig(os.path.join(savefolder, f'{id}.png'), bbox_inches='tight')
        print(f'Image {id} saved in {savefolder}')
    plt.close()


def demo_mist(images_mist, images_no_mist, n_images=None, savefolder=None):
    """
    create a plot of side by side images with and without mist, for demonstration
    purposes
    """
    n_images = len(images_mist) if n_images is None else n_images

    fig = plt.figure(figsize=(n_images*5, 8))
    gs = GridSpec(nrows=2, ncols=n_images, wspace=0.05)
    for i in range(n_images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images_no_mist[i])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        #
        ax = fig.add_subplot(gs[1, i])
        ax.imshow(images_mist[i])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    if savefolder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savefolder, 'mist_demo.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savefolder, 'mist_demo.pdf'), bbox_inches='tight')
    plt.close()


def demo_discarded(images, coverage=0, savefolder=None):
    """
    plots a number of discarded images for demonstration purposes
    INPUTS
    @images: numpy array of the discarded images
    @coverage: float in [0,1] denoting what percentage of the image should be covered
                by bridge
    @save_folder: folder to save the image in
    """
    n_images = min(5, len(images))
    fig = plt.figure(figsize=(n_images*5, 6))
    gs = GridSpec(nrows=1, ncols=n_images)
    for i in range(n_images):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(images[i])
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('Discarded images\n(image coverage below {}%)'.format(int(coverage*100)))
    if savefolder is None:
        plt.show()
    else:
        plt.savefig(os.path.join(savefolder, 'discarded_examples.png'), bbox_inches='tight')
        plt.savefig(os.path.join(savefolder, 'discarded_examples.pdf'), bbox_inches='tight')
    plt.close()


def plot_training_history(hist, save_path=None):
    figure = plt.figure()
    plt.plot(hist['loss'], label='training loss')
    plt.plot(hist['val_loss'], label='validation loss')
    plt.legend()
    if save_path is None:
        plt.imshow()
    else:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(confusion_matrix, labels=None, savepath=None):
    labels = list(range(confusion_matrix.max())) if labels is None else labels
    fig = plt.figure(figsize=(12,12))
    ax = plt.axes()
    # ax.set_title('True VS predicted class')
    ax.grid(False)
    cm = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix,
                display_labels=labels)
    cm.plot(ax=ax, cmap='cividis') #'magma'
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()


def tfonnx_comparison(img, tf_pred, onnx_pred, savepath='./tf_onnx_comparison.png'):
    cmap = LinearSegmentedColormap.from_list('cm', class_colors, N=6)
    fig = plt.figure(figsize=(18,8))
    gs = GridSpec(nrows=1, ncols=4)
    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_rgb.imshow(img)
    ax_rgb.set_title('Original image')
    ax_rgb.grid(False)
    ax_rgb.set_xticks([])
    ax_rgb.set_yticks([])
    #
    ax_tf = fig.add_subplot(gs[0, 1])
    ax_tf.imshow(tf_pred, vmin=0, vmax=5, cmap=cmap)
    ax_tf.set_title('Tensorflow prediction')
    ax_tf.grid(False)
    ax_tf.set_xticks([])
    ax_tf.set_yticks([])
    #
    ax_onnx = fig.add_subplot(gs[0, 2])
    ax_onnx.imshow(onnx_pred, vmin=0, vmax=5, cmap=cmap)
    ax_onnx.set_title('ONNX prediction')
    ax_onnx.grid(False)
    ax_onnx.set_xticks([])
    ax_onnx.set_yticks([])
    #
    ax_diff = fig.add_subplot(gs[0, 3])
    img_diff = onnx_pred != tf_pred
    img_diff = img_diff.astype(int)
    ax_diff.imshow(img_diff, vmin=0, vmax=1)
    ax_diff.set_title('Points of difference')
    ax_diff.grid(False)
    ax_diff.set_xticks([])
    ax_diff.set_yticks([])
    if savepath is None:
        plt.show()
    else:
        plt.savefig(savepath, bbox_inches='tight')
    plt.close()
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
