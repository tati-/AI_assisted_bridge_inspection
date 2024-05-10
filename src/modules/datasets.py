import os
import re
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

from . import utils
from .constants import LABELS_PIPO


def organize_sample_paths(image_paths, mask_paths, savefile=None):
    """
    This function, given a list of image and mask paths, creates a pandas dataframe
    that contains a column for the image paths, and one column per
    label. Each row corresponds to a sample, and holds the paths for this sample's
    image and masks.
    Before loading the data a check is performed, and only the data for which
    both an image and at least one label is available are included in the dataframe.
    This is checked based on the ids that are implied in the file naming (image_<id>).
    If more than one data samples with the same ids exist, this sample is not included
    """
    labels = LABELS_PIPO[1:]
    # infer sample id from the image name
    ids = [os.path.splitext(os.path.basename(x))[0] for x in image_paths]
    ids = [id.replace('image_', '') for id in ids]
    paths = {key: list() for key in ['image']+labels}
    for i, id in enumerate(ids):
        img_paths = [p for p in image_paths if re.search(f'\{os.sep}(image_)?{id}\.\w{{2,5}}$', p)]
        if len(img_paths)!=1:
            # ignore sample if more than one samples with the same id exist
            continue
        else:
            paths['image'].append(img_paths[0])
        for label in labels:
            # find mask paths that correspond to the current id
            pattern = f'\{os.sep}{label}\{os.sep}(mask_)?{id}\.\w{{2,5}}$'
            corr_masks = [p for p in mask_paths if re.search(pattern, p)]
            if len(set(corr_masks))==1:
                paths[label].append(corr_masks[0])
            else:
                paths[label].append(None)
    df = pd.DataFrame.from_dict(paths)
    # delete rows with no labels if they exist
    df.dropna(axis=0, how='all', subset=labels, inplace=True)

    if savefile is not None:
        df.to_csv(savefile)

    return df

################################################################################
# DATASET CLASS
################################################################################
class PFBridgeDataset(tf.keras.utils.Sequence):
    """
    """
    def __init__(self,
                root_path: str,
                batch_size: int=8,
                n_batches: int=None,
                img_size: tuple=None,
                shuffle: bool=True,
                **kwargs
                ):
        """
        INPUTS:
        @root_path: directory where the data infomation is stored. An images and a
                    masks folder is expected inside the root directory
        @batch_size
        @n_batches: total number of batches to be used before the iterator stops
                    (one training epoch)
        @img_size: (width, height) = (ncols, nrows) of desired images
        @shuffle: boolean
        """
        super().__init__(**kwargs)
        self.root_path = root_path
        files = utils.files_with_extensions('png', 'jpg', 'JPG',
                                                datapath=root_path, recursive=True)
        img_paths = [x for x in files if f'{os.sep}images{os.sep}' in x]
        mask_paths = [x for x in files if f'{os.sep}masks{os.sep}' in x]
        # self.labels = list(mask_paths.keys())
        # if 'background' not in labels:
        #     self.labels = ['background'] + self.labels
        self.data_info = organize_sample_paths(img_paths, mask_paths)
        self.n_samples = len(self.data_info)
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_batches = n_batches if n_batches is not None else self.n_samples//self.batch_size
        # avoid seeing the same sample twice in an epoch
        self.n_batches = min(self.n_batches, self.n_samples//self.batch_size)
        self.sample_indices = np.arange(self.n_samples)
        self.epoch_batch_indices = self._epoch_batch_indices()


    def __len__(self):
        """
        number of batches the generator can produce
        """
        return self.n_batches


    def _epoch_batch_indices(self):
        """
        returns a n_batches x batch_size array with the indices for each
        batch for an epoch
        """
        if self.shuffle:
            np.random.shuffle(self.sample_indices)

        epoch_indices = self.sample_indices[:self.n_batches*self.batch_size]
        epoch_batch_indices = np.reshape(epoch_indices,
                                (self.n_batches, self.batch_size))

        return epoch_batch_indices


    def sample_loader(self, sample_ind: int) -> tuple:
        """
        loads a single sample from the dataset, returns an
        (image, masks) tuple
        """
        sample_info = self.data_info.iloc[sample_ind]
        #load image
        image = cv2.resize(
                    cv2.cvtColor(
                        cv2.imread(sample_info.image), cv2.COLOR_BGR2RGB
                        ),
                    self.img_size)
        if np.sum(image) == 0:
            # reject black images
            return None
        else:
            # make sure image is in the 0-1 range
            image = image/(np.max(image))
        # load masks
        masks = np.zeros((image.shape[:-1] + (len(LABELS_PIPO),)))
        for l, label in enumerate(LABELS_PIPO[1:]):
            try:
                mask = cv2.resize(
                            cv2.imread(sample_info[label], cv2.IMREAD_GRAYSCALE),
                            self.img_size)
                # imread can give also intermediate grey values
                mask = np.round(mask/(np.max(mask)+1e-5))
                masks[..., l+1] = mask
            except:
                # if a mask file does not exist or cannot be loaded, the mask is left
                # black for the moment
                continue
        else:
            return image, masks


    def __getitem__(self, batch_ind: int) -> tuple:
        """
        generates a batch of data
        """
        images = []
        masks = []
        for ind in self.epoch_batch_indices[batch_ind]:
            sample = self.sample_loader(ind)
            if sample is not None:
                images.append(sample[0])
                masks.append(sample[1])
        return np.asarray(images), np.asarray(masks)


    def on_epoch_end(self):
        """
        resets the batch indices sets
        """
        self.sample_indices = np.arange(self.n_samples)
        self.epoch_batch_indices = self._epoch_batch_indices()
