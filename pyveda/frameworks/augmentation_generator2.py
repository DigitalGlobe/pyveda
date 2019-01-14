import numpy as np

from pyveda.frameworks.transforms import *

class BaseGenerator():
    def __init__(self, cache, shape=None, batch_size=32, shuffle=True, vertical_flip=False,
        random_rotation = False,horizontal_flip = False):
        self.cache = cache
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.random_rotation = random_rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.on_epoch_end()

    def build_batch(self, index):
        raise NotImplemented

    def __getitem__(self, index):
        # yield self.build_batch(index)
        return self.build_batch(index)

    def __next__(self):
        try:
            item = self[self.index]  # index???
        except IndexError:
            raise StopIteration
        self.index += 1
        return item

    def _process(self, random_rotation, horizontal_flip, vertical_flip):
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
    def __init__(self, cache, batch_size):
        super().__init__(cache, batch_size=batch_size, shuffle=True)
        self.list_ids = [i for i in range(0, len(self.cache))]
        self.mltype = cache._trainer.mltype
        self.shape = cache._trainer.image_shape

    def build_batch(self, index):
        '''Generate one batch of data'''
        if index > len(self):
            raise IndexError("index is invalid")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        # print(list_ids_temp)
        X, y = self._data_generation(list_ids_temp)
        # print("build vb batch")
        return X, y

    def _data_generation(self, list_ids_temp):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        X = np.empty((self.batch_size, *self.shape[::-1]))  # issue with shape?
        if self.mltype == 'classification':
            y = np.empty((self.batch_size), dtype=int)   # needs classes
        if self.mltype == 'segmentation':
            y = np.empty((self.batch_size, *self.shape[1:]))   # good
        if self.mltype == 'object_detection':
            y = []

        for i, _id in enumerate(list_ids_temp):
            X[i, ] = self.cache.images[_id].T
            if self.mltype == 'classification':
                y[i, ] = self.cache.labels[_id]
            if self.mltype == 'object_detection':
                y.append(self.cache.labels[_id])
            if self.mltype == 'segmentation':
                #  will need to adjust based on augmentation (flipping/rotation)
                y[i, ] = self.cache.labels[_id]
        if self.mltype == 'object_detection':
            return X, np.array(y)
        else:
            return X, y


class VedaStreamGenerator(BaseGenerator):
    '''
    VedaStream
    '''
    def __init__(self):
            # self.mltype = cache._vset.mltype
            # self.shape = cache._vset.image_shape
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()
    # '''Denotes the number of batches per epoch'''
    #     return int(np.floor(self.allocated)/self.batch_size)

    def build_batch(self):
        raise NotImplementedError()
