import os
import json


class Release():
    ''' Access to file-based Releases '''

    def init(self, source):
        self.source = source
        images = os.path.join(self.source, 'images')
        labels = os.path.join(self.source, 'labels')
        meta = os.path.join(self.source, 'meta.json')
        bounds = os.path.join(self.source, 'bounds.json')
        if not (os.path.exists(images) or os.path.exists(
                labels) or os.path.exists(meta)):
            raise ValueError('{} is not a valid release'.format(self.source))
        with open(meta, 'r') as meta_file:
            self.meta = json.load(meta_file)
        try:
            with open(bounds, 'r') as bounds_file:
                self.bounds = json.load(bounds_file)
                self.__geo_interface__ = self.bounds
        except BaseException:
            # the release stores ungeoreferenced imagery
            self.bounds = None

    def store(self, path, size):
        from dataset import DataSet
        ''' Stores the Release as an hdf5 Dataset '''
        ds = DataSet(self, size=size)
        return ds.store(path)
