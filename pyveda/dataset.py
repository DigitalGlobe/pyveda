import os
import json
from training import VedaCollection

class DataSet():

    def __init__(self, source, size=None):
        self.size = size
        self.source = source
        self.data = None
        if type(source) == str:
            if os.path.splitext(source)[1] == 'hd5':
                self.data_type = 'hdf5'
                #load
            else:
                from release import Release
                release = Release(source)
                self.source = release
        if type(self.source) == Release:
            self.data_type = 'memory'
            #load
        elif type(self.source) == VedaCollection:
            self.data_type = 'memory'
            #load
    
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
    def classes(self):
        ''' A list of all the classes, in the same order as the labels in the `labels` array. 
            Classes and labels are always in alphabetical order by class name.'''
        return self.data.classes
    
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