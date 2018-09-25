# Based on the implementation of git user bernardohenz (Bernardo Henz).
# Link: https://github.com/bernardohenz/ExtendableImageDatagen

import numpy as np
import keras
from keras_preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage as ndi
from random import randint
from random import sample

# Pre-processing Functions

# Jamie's rescaling function


def rescale_toa(arr, dtype=np.float32):
    """
    Rescale any multi-dimensional array of shape (D, M, N) by first subtracting
    the min from each (M, N) array along axis 0, then divide each by the
    resulting maximum to get (float32) distributions between [0, 1].
    Optionally spec uint8 for 0-255 ints.
    """
    # First look at raw value dists along bands

    arr_trans = np.subtract(arr, arr.min(axis=(1, 2))[:, np.newaxis, np.newaxis])
    arr_rs = np.divide(arr_trans, arr_trans.max(axis=(1, 2))[:, np.newaxis, np.newaxis])
    if dtype == np.uint8:
        arr_rs = np.array(arr_rs*255, dtype=np.uint8)
    return arr_rs.T


def bands_subset_f(arr, band_numbers):
    """
    Subset array to array of  just given band indices
    arr: float32
    band_numbers: List. List of band numbers to subset
    """
    return arr[band_numbers, ...]

# Augmentation Functions Definition- written by Bernardo Henz
# https://github.com/bernardohenz/ExtendableImageDatagen/blob/master/util/extendable_datagen.py


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel,
                      final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def random_rotation_f(x, rg, row_index=1, col_index=2, channel_index=0,
                      fill_mode='nearest', cval=0.):
    theta = np.pi / 180 * np.random.uniform(-rg, rg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                [np.sin(theta), np.cos(theta), 0],
                                [0, 0, 1]])

    h, w = x.shape[row_index], x.shape[col_index]
    transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
    return x


class DataGenerator(keras.utils.Sequence):
    '''
    cache:
    batch_size: Int.
    shape: Tuple. (number of bands - after data has been preprocessed bands, height, width)
    group: String. train, test, or validate.
    rescale_toa: Boolean. To rescale values between 0 and 1.
    bands_subset: List. List of band numbers to subset.
    random_rotation: Boolean. Randomly rotate image by selected degree.
    horizontal_flip: Boolean. Randomly flip inputs horizontally.
    vertical_flip: Boolean. Randomly flip inputs vertically.
    '''

    def __init__(self, cache, batch_size, shape, shuffle=True, group="train",
                 rescale_toa=False, bands_subset=None,
                 random_rotation=False, horizontal_flip=False,
                 vertical_flip=False):

        self.cache = getattr(cache, group)
        self.batch_size = batch_size
        self.list_ids = [i for i in range(0, len(self.cache))]
        self.shuffle = shuffle
        self.on_epoch_end()
        self.group = group
        self.rescale_toa = rescale_toa
        self.bands_subset = bands_subset
        self.random_rotation = random_rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def process(self, random_rotation, horizontal_flip, vertical_flip):
        augmentation_list = []
        if self.random_rotation:
            augmentation_list.append(random_rotation_f)
        if self.horizontal_flip:
            augmentation_list.append(np.fliplr)
        if self.vertical_flip:
            augmentation_list.append(np.flipud)
        return augmentation_list

    def on_epoch_end(self):
        '''update index for each epoch'''
        self.indexes = np.arange(len(self.list_ids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_ids_temp):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        X = np.empty((self.batch_size, *self.shape[::-1]))
        y = np.empty((self.batch_size), dtype=int)

        augmentation_lst = self.process(self.random_rotation,
                                        self.horizontal_flip,
                                        self.vertical_flip)

        for i, _id in enumerate(list_ids_temp):
            if self.rescale_toa and self.bands_subset is not None:
                x = rescale_toa(bands_subset_f(self.cache.image[_id], self.bands_subset))
            if self.bands_subset is not None and not self.rescale_toa:
                x = bands_subset_f(self.cache.image[_id], self.bands_subset).T
            if self.rescale_toa is True and self.bands_subset is None:
                x = rescale_toa(self.cache.image[_id])
            if self.bands_subset is None and not self.rescale_toa:
                x = self.cache.image[_id].T

            # user selects no augmentation functions
            if len(augmentation_lst) == 0:
                X[i, ] = x

            else:
                randomly_selected_functions_lst = sample(augmentation_lst, randint(0, len(augmentation_lst) - 1))
                # possibility that no augmentation functions were selected
                if len(randomly_selected_functions_lst) == 0:
                    X[i, ] = x
                for func in randomly_selected_functions_lst:
                    if func is random_rotation_f:
                        random_rotation = randint(0, 359)
                        x = func(x, random_rotation)
                    else:
                        x = func(x)
                X[i, ] = x
            y[i] = self.cache.classification[_id]
        return X, y

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.list_ids) / self.batch_size))

    def __getitem__(self, index):
        '''Generate one batch of data'''
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        X, y = self.data_generation(list_ids_temp)
        return X, y
