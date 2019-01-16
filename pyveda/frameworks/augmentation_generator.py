import numpy as np

from random import randint
from random import sample

from pyveda.frameworks.transforms import *


class BaseGenerator():
    def __init__(self, cache, shape=None, batch_size=32, shuffle=True, rescale=False):
        self.cache = cache
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.rescale = rescale
        self.on_epoch_end()
        self.list_ids = np.arange(0, len(self.cache))

    def build_batch(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        return self.build_batch(index)

    def __next__(self):
        try:
            item = self[self.index]  # index???
        except IndexError:
            raise StopIteration
        self.index += 1
        return item

    def on_epoch_end(self):
        '''update index for each epoch'''
        # self.indexes = np.arange(len(self.list_ids)) self.list_ids not defined in baseclass
        self.indexes = np.arange(len(self.cache))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __iter__(self):
        """Create a generator that iterates over the Sequence."""
        for item in (self[i] for i in range(len(self))):
            self.index += 1
            yield item

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.cache)/self.batch_size))


class VedaStoreGenerator(BaseGenerator):
    '''
    VedaBase
    '''

    @property
    def mltype(self):
        return self.cache._trainer.mltype

    @property
    def shape(self):
        return self.cache._trainer.image_shape

    def build_batch(self, index):
        '''Generate one batch of data'''
        if index > len(self):
            raise IndexError("Batch generator is exhausted, index is invalid.")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        x, y = self._data_generation(list_ids_temp)
        return x, y

    def _data_generation(self, list_ids_temp):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        #x = np.empty((self.batch_size, *self.shape[::-1]))
        x = np.empty((self.batch_size, *self.shape))  # issue with shape?
        if self.mltype == 'classification':
            y = np.empty((self.batch_size), dtype=int)   # needs classes
        if self.mltype == 'segmentation':
            y = np.empty((self.batch_size, *self.shape[1:]))
        if self.mltype == 'object_detection':
            y = []

        for i, _id in enumerate(list_ids_temp):

            x_img = self.cache.images[_id]
            if self.rescale:
                x_img = rescale_toa(x_img)
            x[i, ] = x_img
            if self.mltype == 'classification':
                y[i, ] = self.cache.labels[_id]
            if self.mltype == 'object_detection':
                y.append(self.cache.labels[_id])
            if self.mltype == 'segmentation':
                y[i, ] = self.cache.labels[_id]
        if self.mltype == 'object_detection':  # indent level?
            return x, np.array(y)
        else:
            return x, y
