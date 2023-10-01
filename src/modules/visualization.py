import os
import sys
import pdb
import glob
import time
import random
import datetime
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
sns.set(font='DejaVu Serif', font_scale=2)
sns.set_style('darkgrid', {'font.family':'serif'})

import modules.dataset_utils as dts
from .decorators import forall


# get values of preexisting colormap as list
class_colors = cm.get_cmap('tab20')((np.linspace(0, 1, 12)))[:, :3]
class_colors[0, :] = np.asarray([0,0,0])


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
def inspect_dataset(paths, labels, savefolder=None):
    """
    this function displays each data sample image with its groundtruth and
    potentially with the network prediction.
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
    img_path = paths.image
    mask_paths = {key: [paths[key]] for key in paths.keys()[1:]}

    im, gt, _ = dts.data_loader([img_path], mask_paths, labels=labels)
    # get number of pixel labels per class
    class_concentr = np.sum(gt, axis=tuple(range(gt.ndim-1)))
    class_concentr = {label: int(class_concentr[i]) for i,label in enumerate(labels)}
    #
    im, gt = im[0], np.argmax(gt[0], axis=-1)
    im_id = os.path.splitext(os.path.basename(paths.image))[0].replace('image_', '')
    if im_id=='':
        pdb.set_trace()
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
