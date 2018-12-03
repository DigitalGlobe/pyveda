import numpy as np
import dask.array as da
from .loaders import load_image
from shapely.geometry import shape as shp, mapping, box
from shapely import ops
import requests

from pyveda.auth import Auth
gbdx = Auth()
conn = gbdx.gbdx_connection

class DataPoint(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), dtype='uint8', **kwargs):
        self.conn = conn
        self.links = item["properties"].get("links")
        self.imshape = list(shape)
        self._y = None
        self.data = item['properties']
        self._dtype = dtype
        self.classes = kwargs.get('classes')

    @property
    def id(self):
        return self.data["id"]

    @property
    def mltype(self):
        try:
            return self.data['mltype']
        except KeyError:
            return self.data.get('mlType')

    @property
        if 'dtype' in self.data:
            return np.dtype(self.data['dtype'])
        return np.dtype(self._dtype)

    @property
    def label(self):
        return self.data['label']

    @property
    def dataset_id(self):
        return self.data['dataset_id']

    @property
    def bounds(self):
        return self.data.get("bounds")

    @property
    def tile_coords(self):
        return self.data.get("tile_coords")

    @property
    def y(self):
        if self._y is None:
            self._y = self._map_labels()
        return self._y

    @property
    def image(self):
        """ Returns a delayed dask call for fetching the image for a data point """
        token = gbdx.gbdx_connection.access_token
        load = load_image(self.links["image"]["href"], token, self.imshape, dtype=self.dtype)
        return da.from_delayed(load, shape=self.imshape, dtype=self.dtype)

    def save(self, data):
        """ Saves/updates the datapoint in the database """
        return self.conn.put(self.links["update"]["href"], json=data).json()

    def update(self, new_data, save=True):
        """ Updates data for the datapoint in the database """
        self.data.update(new_data)
        if save:
           self.save(new_data)

    def remove(self):
        """ Removes the datapoint from the set"""
        self.conn.delete(self.links["delete"]["href"])

    def _map_labels(self):
        """ Convert labels to data """
        if self.mltype:
            if self.mltype == 'classification':
                return [int(self.label[key]) for key in list(self.label.keys())]
            return self.label

    def __repr__(self):
        data = self.data.copy()
        del data['links']
        return str(data)

    @property
    def __geo_interface__(self):
        return box(*self.bounds).__geo_interface__
