import numpy as np
import math
from random import choice

from pyveda.frameworks.transforms import *


def transform_b(revised_y_lst):
    for i in np.arange(len(revised_y_lst)):
        for x in np.arange(len(revised_y_lst[i])):
            revised_y_lst[i][x] = np.asarray(revised_y_lst[i][x])
    return np.asarray(revised_y_lst)

class BaseGenerator():
    '''
    Parent Class for Generator

    cache (VedaBase or VedaStream partition): Partition (train, test, or validate)
    batch_size (int): Number of samples in batch
    steps (int): Nnumber of steps of batches to run in one epoch. If not provided, will calculate maximum possible number of complete batches
    loop (boolean): if true, batcher will loop infinitely, if false, StopIteration is thrown after 1 epoch
    shuffle (boolean): To shuffle or not shuffle data between epochs.
    channels_last: Boolean. To return image data as Height-Width-Depth, instead of the default Depth-Height-Width
    Rescale: Boolean. Flag to indicate if data returned from the generator should be rescaled between 0 and 1.
    flip_horizontal: boolean. Horizontally flip image and lables.
    flip_vertical: boolean. Vertically flip image and lables
    pad: Int. New larger dimension to transform image into.
    '''

    def __init__(self, cache, batch_size=32, steps=None, loop=True, shuffle=True, channels_last=False, expand_dims=False, rescale=False,
                    flip_horizontal=False, flip_vertical=False, pad=None, label_transform=None, batch_label_transform=None,
                    image_transform=None):
        self.cache = cache
        self.batch_size = batch_size
        self.epoch = 1
        # steps start at 1
        self.step = 1
        self.last_step = int(math.floor(len(self.cache)/self.batch_size))
        if steps is not None:
            assert steps <= self.last_step, "{} datapoints is not enough for {} steps with a batch size of {}".format(len(self.cache), steps, self.batch_size)
            self.last_step = steps
        self.loop = loop
        self.id_lut = np.arange(len(self.cache))
        self.shuffle = shuffle
        if self.shuffle:
            self.shuffle_ids()
        self.channels_last = channels_last
        self.expand_dims = expand_dims
        self.rescale = rescale
        self.flip_h = flip_horizontal
        self.flip_v = flip_vertical
        self.pad = pad
        self.label_transform = label_transform
        self.batch_label_transform = batch_label_transform
        self.image_transform = image_transform

        if label_transform is not None:
            assert callable(label_transform), "label_transform must be a method"

        if batch_label_transform is not None:
            assert callable(batch_label_transform), "batch_label_transform must be a method"

        if image_transform is not None:
            assert callable(image_transform), "image_transform must be a method"

        if pad is not None:
            assert isinstance(pad, int), "Pad must be an integer"
            assert max(self.shape) < pad, "Pad size must be larger than image size"

    @property
    def mltype(self):
        return self.cache._vset.mltype

    @property
    def shape(self):
        return self.cache._vset.image_shape

    def shuffle_ids(self):
        '''update index for each epoch'''
        np.random.shuffle(self.id_lut)

    def _applied_augs(self):
        """
        Returns randomly selected augmentation functions from a list of eligible augmentation functions
        """
        augmentation_list = []
        if self.flip_h and choice([True, False]):
            augmentation_list.append(np.fliplr)
        if self.flip_v and choice([True, False]):
            augmentation_list.append(np.flipud)
        return augmentation_list

    def apply_augmentations(self, x, y):
        """
            Applies all built-in transforms, returns augmented x/y
            Args:
              x: image array
              y: label
        """

        # Make Augmentations
        for t_fn in self._applied_augs():
            x = t_fn(x)

            if self.mltype == 'segmentation':
                y = t_fn(y)

            elif self.mltype == 'object_detection':
                if t_fn is np.fliplr:
                    y = flip_labels_horizontal(self.shape, y)
                elif t_fn is np.flipud:
                    y = flip_labels_vertical(self.shape, y)
        
        return x, y

    def build_batch(self, step):
        raise NotImplementedError

    def __getitem__(self, batch):
        raise NotImplementedError("Getting a specific batch is not supported since batches are generally randomized")

    def __next__(self):
        if self.step > self.last_step:
            if not self.loop:
                raise StopIteration
            else:
                self.epoch += 1
                self.step = 1
                if self.shuffle:
                    self.shuffle_ids()
        batch = self.build_batch(self.step)
        self.step += 1
        return batch

    def __iter__(self):
        """Create a generator that iterates over the Sequence."""
        return self

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return self.last_step 


class VedaStoreGenerator(BaseGenerator):
    '''
    Generator for VedaBase partition, either train, test or validate.
    '''

    def build_batch(self, step):
        '''Generate one batch of data for a given step'''
        start = (step-1) * self.batch_size
        end = step * self.batch_size
        batch_ids = self.id_lut[start:end]
        x, y = self._data_generation(batch_ids)
        return x, y

    def _data_generation(self, batch_ids):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        # setup empty batch
        if self.channels_last:
            x = np.empty((self.batch_size, *self.shape[::-1]))
        else:
            x = np.empty((self.batch_size, *self.shape))

        y = []

        for i, _id in enumerate(batch_ids):
            x_img = self.cache.images[int(_id)]
            y_img = self.cache.labels[int(_id)]

            x_img, y_img = self.apply_augmentations(x_img, y_img)

            if self.channels_last:
                x_img = x_img.T

            if self.image_transform:
                x_img = self.image_transform(x_img)

            if self.expand_dims:
                y_img = np.expand_dims(y_img, 2)

            if self.label_transform:
                y_img = [self.label_transform((_y, indx)) for indx, _x in enumerate(y_img) for _y in _x]

            x[i, ] = x_img
            y.append(y_img)

        if self.rescale:
            x /= x.max()

        if self.pad:
            x = pad(x, self.pad, self.channels_last)

        if self.batch_label_transform:
            t = [np.asarray(i) for i in y]
            y = self.batch_label_transform(t)
            
        return x, np.array(y) 

class VedaStreamGenerator(BaseGenerator):
    '''
    Generator for VedaStream parition, either train, test, or validate.
    '''

    def build_batch(self, step):
        '''Generate one batch of data'''
        x, y = self._data_generation()
        return x, y

    def _data_generation(self):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        if self.channels_last:
            x = np.empty((self.batch_size, *self.shape[::-1]))
        else:
            x = np.empty((self.batch_size, *self.shape))
        y = []

        for i in range(self.batch_size):
            x_img, y_img = next(self.cache)
            x_img, y_img = self.apply_augmentations(x_img, y_img)

            if self.channels_last:
                x_img = x_img.T

            if self.image_transform:
                x_img = self.image_transform(x_img)

            if self.expand_dims:
                y_img = np.expand_dims(y_img, 2)

            if self.label_transform:
                y_img = [self.label_transform((_y, indx)) for indx, _x in enumerate(y_img) for _y in _x]

            x[i, ] = x_img
            y.append(y_img)

        if self.rescale:
            x /= x.max()

        if self.pad:
            x = pad(x, self.pad, self.channels_last)

        if self.batch_label_transform:
            t = [np.asarray(i) for i in y]
            y = self.batch_label_transform(t)

        return x, np.array(y)
