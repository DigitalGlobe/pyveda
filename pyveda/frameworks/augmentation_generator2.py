import numpy as np

from random import randint
from random import sample

from pyveda.frameworks.transforms import *


class BaseGenerator():
    def __init__(self, cache, shape=None, batch_size=32, shuffle=True, rescale_toa=False, random_rotation=False,
                 horizontal_flip=False, vertical_flip=False):
        self.cache = cache
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.rescale_toa = rescale_toa
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

    def process(self):
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
    def __init__(self, cache, batch_size, rescale_toa=False, random_rotation=False, vertical_flip=False,
                 horizontal_flip=False):
        super().__init__(cache, batch_size=batch_size, shuffle=True, rescale_toa=rescale_toa, random_rotation=random_rotation,
                         vertical_flip=vertical_flip, horizontal_flip=horizontal_flip)
        self.list_ids = np.arange(0, len(self.cache))
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
            y = np.empty((self.batch_size, *self.shape[1:]))
        if self.mltype == 'object_detection':
            y = []

        augmentation_lst = self.process()

        for i, _id in enumerate(list_ids_temp):
            if self.rescale_toa:
                x = rescale_toa(self.cache.images[_id])
            else:
                x = self.cache.images[_id].T
            if len(augmentation_lst) == 0:
                X[i, ] = x
            else:
                randomly_selected_functions_lst = sample(augmentation_lst, randint(0, len(augmentation_lst) - 1))
                #print(randomly_selected_functions_lst)
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
            if self.mltype == 'classification':
                y[i, ] = self.cache.labels[_id]

            if self.mltype == 'object_detection':
                y.append(self.cache.labels[_id])
            # if self.mltype == 'object_detection' and len(augmentation_lst) == 0:
            #     y.append(self.cache.labels[_id])
            # else:
            #     for func in randomly_selected_functions_lst:
            #         if func is random_rotation_f:
            #             y_od = self.cache.labels[_id] #need to fix
            #         if func is np.fliplr:
            #             y_od = flip_labels_horizontal(self.shape, self.cache.labels[_id])
            #             print(y_od)
            #
            #         if func is np.flipud:
            #             y_od = flip_labels_vertical(self.shape, self.cache.labels[_id])
            #             print(y_od)
            #     #  will need to adjust based on augmentation (flipping/rotation)
            #     y.append(y_od)
            if self.mltype == 'segmentation':
                y[i, ] = y
            # if self.mltype == 'segmentation' and len(augmentation_lst) == 0:
            #     y[i, ] = self.cache.labels[_id]
            # else:
            #     for func in randomly_selected_functions_lst:
            #         if func is random_rotation_f:
            #             random_rotation = randint(0, 359)
            #             y = func(self.cache.labels[_id], random_rotation)
            #         else:
            #             y = func(self.cache.labels[_id])
                # y[i, ] = y
        if self.mltype == 'object_detection':  # indent level?
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
