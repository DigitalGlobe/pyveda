import numpy as np

from random import randint, sample

from pyveda.frameworks.transforms import *


def transform_b(revised_y_lst):
    for i in np.arange(len(revised_y_lst)):
        for x in np.arange(len(revised_y_lst[i])):
            revised_y_lst[i][x] = np.asarray(revised_y_lst[i][x])
    return np.asarray(revised_y_lst)

class BaseGenerator():
    '''
    Parent Class for Generator

    cache: Parition (train, test, or validate) of VedaBase or VedaStream.
    batch_size: int. Number of samples.
    shuffle: Boolean. To shuffle or not shuffle data between epochs.
    channels_last: Boolean. To return image data as Height-Width-Depth, instead of the default Depth-Height-Width
    Rescale: Boolean. Flag to indicate if data returned from the generator should be rescaled between 0 and 1.
    flip_horizontal: boolean. Horizontally flip image and lables.
    flip_vertical: boolean. Vertically flip image and lables
    '''

    def __init__(self, cache, batch_size=32, shuffle=True, channels_last=False, expand_dims=False, rescale=False,
                    flip_horizontal=False, flip_vertical=False, custom_label_transform=None, custom_batch_transform=None,
                    custom_image_transform=None):
        self.cache = cache
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.channels_last = channels_last
        self.expand_dims = expand_dims
        self.rescale = rescale
        self.on_epoch_end()
        self.flip_h = flip_horizontal
        self.flip_v = flip_vertical
        self.list_ids = np.arange(len(self.cache))
        self.custom_label_transform = custom_label_transform
        self.custom_batch_transform = custom_batch_transform
        self.custom_image_transform = custom_image_transform

        if custom_label_transform is not None:
            assert callable(custom_label_transform), "custom_label_transform must be a method"

        if custom_batch_transform is not None:
            assert callable(custom_batch_transform), "custom_batch_transform must be a method"

        if custom_image_transform is not None:
            assert callable(custom_image_transform), "custom_image_transform must be a method"

    @property
    def mltype(self):
        return self.cache._vset.mltype

    @property
    def shape(self):
        return self.cache._vset.image_shape

    def _applied_augs(self, flip_h, flip_v):
        """
        Returns randomly selected augmentation functions from a list of eligible augmentation functions
        """
        augmentation_list = []
        if self.flip_h:
            augmentation_list.append(np.fliplr)
        if self.flip_v:
            augmentation_list.append(np.flipud)

        if len(augmentation_list) == 0:
            return augmentation_list
        else:
            # randomly select augmentations to perform
            randomly_selected_functions_lst = sample(augmentation_list, k=randint(0, len(augmentation_list)))
        return randomly_selected_functions_lst

    def apply_transforms(self, x, y):
        """
            Applies a transform method, returns x/y augmentations
            Args:
              x: image array
              y: label
        """
        # Figure out which transform to apply randomly
        transforms = self._applied_augs(self.flip_h, self.flip_v)

        # User selects no augmentation functions
        if len(transforms) == 0:
                if self.mltype == 'object_detection':
                    return x, y
                return x, y

        # Make Augmentations
        if self.mltype == 'classification':
            for t_fn in transforms:
                x = t_fn(x)
            return x, y

        if self.mltype == 'object_detection':
            for t_fn in transforms:
                x = t_fn(x)
                if t_fn is np.fliplr:
                    y = flip_labels_horizontal(self.shape, y)
                if t_fn is np.flipud:
                    y = flip_labels_vertical(self.shape, y)
            return x, y

        if self.mltype == 'segmentation':
            for t_fn in transforms:
                x = t_fn(x)
                y = t_fn(y)
            return x, y

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
    Generator for VedaBase partition, either train, test or validate.
    '''

    def build_batch(self, index):
        '''Generate one batch of data'''
        if index > len(self):
            raise IndexError("Index is invalid because batch generator is exhausted.")
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_ids_temp = [self.list_ids[k] for k in indexes]
        x, y = self._data_generation(list_ids_temp)
        return x, y

    def _data_generation(self, list_ids_temp):
        '''Generates data containing batch_size samples
        optionally pre-processes the data'''

        # setup empty batch
        if self.channels_last:
            x = np.empty((self.batch_size, *self.shape[::-1]))
        else:
            x = np.empty((self.batch_size, *self.shape))
        y = []

        for i, _id in enumerate(list_ids_temp):
            x_img = self.cache.images[int(_id)]
            y_img = self.cache.labels[int(_id)]
            x_img, y_img = self.apply_transforms(x_img, y_img)
            if self.channels_last:
                x_img = x_img.T
            if self.custom_image_transform:
                x_img = custom_image_transform(x_img)
            x[i, ] = x_img

            if self.expand_dims:
                y_img = np.expand_dims(y_img, 2)

            if self.custom_label_transform: #must be a method
                y_img = [self.custom_label_transform((_y, indx)) for indx, _x in enumerate(y_img) for _y in _x]

            #this is a single y value, apply the transform here?
            y.append(y_img)

        #rescale after entire bactch is collected
        if self.rescale:
            x /= x.max()

        if self.custom_batch_transform:
            print(y[0])
            print(type(y[0]))
            t = [np.asarray(i) for i in y]
            y = self.custom_batch_transform(t)
        return x, np.array(y) #last place we touch y before returned, but this is a whole batch

class VedaStreamGenerator(BaseGenerator):
    '''
    Generator for VedaStream parition, either train, test, or validate.
    '''

    def build_batch(self, index):
        '''Generate one batch of data'''
        if index > len(self):
            raise IndexError("Index is invalid because batch generator is exhausted.")
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

        while len(y) < self.batch_size:
            pair = next(self.cache)
            x_img = pair[0]
            y_img = pair[1]
            x_img, y_img = self.apply_transforms(x_img, y_img)

            if self.channels_last:
                x_img = x_img.T

            if self.custom_image_transform:
                x_img = custom_image_transform(x_img)
            x[len(y), ] = x_img

            if self.expand_dims:
                y_img = np.expand_dims(y_img, 2)
            y.append(y_img)

            if self.custom_label_transform: #must be a method
                y_img = [self.custom_label_transform((_y, indx)) for indx, _x in enumerate(y_img) for _y in _x]
            y.append(y_img)

        if self.rescale:
            x /= x.max()
        if self.custom_batch_transform:
            t = [np.asarray(i) for i in y]
            y = self.custom_batch_transform(t)

        return x, np.array(y)
