import numpy as np


class Process():
    def __init__(self):
        pass
    def rescale_toa(self, arr, dtype=np.float32):
        """
        Rescale any multi-dimensional array of shape (D, M, N) by first subtracting
        the min from each (M, N) array along axis 0, then divide each by the
        resulting maximum to get (float32) distributions between [0, 1].
        Optionally spec uint8 for 0-255 ints.
        """

        arr_trans = np.subtract(arr, arr.min(axis=(1, 2))[:, np.newaxis, np.newaxis])
        arr_rs = np.divide(arr_trans, arr_trans.max(axis=(1, 2))[:, np.newaxis, np.newaxis])
        if dtype == np.uint8:
            arr_rs = np.array(arr_rs*255, dtype=np.uint8)
        return arr_rs.T

    def bands_subset_f(self, arr, band_numbers):
        """
        Subset array to array of  just given band indices
        arr: float32
        band_numbers: List. List of band numbers to subset
        """
        return arr[band_numbers, ...]

    def apply_transform(self, x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
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

    def transform_matrix_offset_center(self, matrix, x, y):
        o_x = float(x) / 2 + 0.5
        o_y = float(y) / 2 + 0.5
        offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
        reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
        transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
        return transform_matrix


    def random_rotation_f(self, x, rg, row_index=1, col_index=2, channel_index=0,
                          fill_mode='nearest', cval=0.):
        theta = np.pi / 180 * np.random.uniform(-rg, rg)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])

        h, w = x.shape[row_index], x.shape[col_index]
        transform_matrix = transform_matrix_offset_center(rotation_matrix, h, w)
        x = apply_transform(x, transform_matrix, channel_index, fill_mode, cval)
        return x


class BaseGenerator():
    def __init__(self, cache, shape=None, batch_size=32, shuffle=True, rescale_toa=False, bands_subset=None,
                random_rotation=False, horizontal_flip=False, vertical_flip=False):
        self.cache = cache
        self.batch_size = batch_size
        self.index = 0
        self.shuffle = shuffle
        self.on_epoch_end()
        self.rescale_toa = rescale_toa
        self.bands_subset = bands_subset
        self.random_rotation = random_rotation
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

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
        super().__init__(cache, batch_size=batch_size, shuffle=True,
                        rescale_toa=False, bands_subset=None,random_rotation=False,
                        horizontal_flip=False, vertical_flip=False)
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
        X, y = self.data_generation(list_ids_temp)
        # print("build vb batch")
        return X, y

    def data_generation(self, list_ids_temp):
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
