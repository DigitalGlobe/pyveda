import os
import json
from .training import VedaCollection
from .release import Release

class DataSet():

    def __init__(self, source, size=None):
        self.size = size
        self.source = source
        self.data = None
        if type(source) == str:
            if os.path.splitext(source)[1] == 'hd5':
                self.data_type = 'hdf5'
                self._load_hdf5()
            else:
                release = Release(source)
                self.source = release
        if type(self.source) == Release:
            self._load_release()
        elif type(self.source) == VedaCollection:
            self._load_vc()

    def _load_release(self):
        self.data_type = 'memory'
        self.classes = self.source.classes
        #load

    def _load_hdf5(self):
        #hydrate the Imagetrainer?
        pass


    def _load_vc(self):
        self.data_type = 'memory'
        self.classes = self.source.classes
        if not self.size:
            self.size = self.source.count
        self.data = self.source.batch(self.size)    

    def store(self, path, size=None):
        ''' Convert the Dataset to HDF5 storage if it's using in-memory'''
        if self.data_type == 'hdf5':
            print('DataSet already saved to HDF5 file at {}'.format(self.source))
            return
        # Do the conversion
        self.source = path
        self.data_type = 'hdf5'
        return self
    
    @property
    def images(self):
        '''Returns a an array-like object (either in-memory or and HDF5 object) of all the images.'''
        return self.data.images

    @property
    def labels(self):
        '''Returns an array-like object (either in-memory or and HDF5 object) of all the labels.'''
        return self.data.labels

    @property
    def train(self):
        self.data = self.data.train
        return self

    @property
    def validate(self):
        self.data = self.data.validate
        return self

    @property
    def test(self):
        self.data = self.data.test
        return self