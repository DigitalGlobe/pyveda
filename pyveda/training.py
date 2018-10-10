import six
import os, shutil, json, math
import time
import sys
import mmap
import random
import h5py
from collections import defaultdict
try:
    from urllib.parse import urlencode
except:
    from urllib import urlencode
import requests
from requests_futures.sessions import FuturesSession
from tempfile import NamedTemporaryFile
import numpy as np
from dask import delayed
import dask.array as da
from gbdxtools import Interface
from .loaders import load_image
from .utils import transforms
from rasterio.features import rasterize
from shapely.geometry import shape as shp, mapping
from shapely import ops

from functools import partial
import dask

import threading

from .hdf5 import ImageTrainer
from pyveda.utils import NamedTemporaryHDF5Generator
from pyveda.fetch.compat import write_fetch

threads = int(os.environ.get('GBDX_THREADS', 64))
threaded_get = partial(dask.threaded.get, num_workers=threads)
gbdx = Interface()

HOST = os.environ.get('SANDMAN_API')
if not HOST:
    HOST = "https://veda-api.geobigdata.io"

if 'https:' in HOST:
    conn = gbdx.gbdx_connection
else:
    headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.access_token)}
    conn = requests.Session()
    conn.headers.update(headers)

valid_mltypes = ['classification', 'object_detection', 'segmentation']

def search(params={}):
    r = conn.post('{}/{}'.format(HOST, "search"), json=params)
    #try:
    results = r.json()
    return [VedaCollection.from_doc(s) for s in r.json()]
    #except Exception as err:
    #    print(err)
    #    return []

def vec_to_raster(vectors, shape):
    try:
        arr = rasterize(((g, 1) for g in vectors), out_shape=shape[1:])
    except Exception as err:
        #print(err)
        arr = np.zeros(shape[1:])
    return np.array(arr)

class DataPoint(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), dtype="uint8", **kwargs):
        self.conn = conn
        self.data = item["data"]
        self.links = item["links"]
        self.shape = tuple(map(int, shape))
        self.dtype = dtype
        self.ml_type = kwargs.get('mlType')
        self._y = None

        if self.ml_type is not None and self.ml_type == 'segmentation':
            self.data['label'] = vec_to_raster(self.data['label'], self.shape)


    @property
    def id(self):
        return self.data["id"]

    @property
    def label(self):
        return self.data['label']

    @property
    def y(self):
        if self._y is None:
            self._y = self._map_labels()
        return self._y

    @property
    def image(self):
        """ Returns a delayed dask call for fetching the image for a data point """
        token = gbdx.gbdx_connection.access_token
        load = load_image(self.links["image"]["href"], token, self.shape, dtype=self.dtype)
        return da.from_delayed(load, shape=self.shape, dtype=self.dtype)

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
        return self.conn.delete(self.links["delete"]["href"]).json()

    def _map_labels(self):
        """ Convert labels to data """
        _type = self.ml_type
        if _type is not None:
            if _type == 'classification':
                return [int(self.label[key]) for key in list(self.label.keys())]
            elif _type == 'segementation':
                return vec_to_raster(self.label, self.shape)
            else:
                return self.label
        else:
            return None

    def __repr__(self):
        return str(self.data)


class BaseSet(object):
    """ Base class for handling all API interactions on sets."""
    def __init__(self, id=None):
        self.id = id
        self._data_url = "{}/data".format(HOST)
        self._cache_url = "{}/data/{}/cache"
        self._datapoint_url = "{}/datapoints".format(HOST)
        self._chunk_size = int(os.environ.get('VEDA_CHUNK_SIZE', 5000))
        self.conn = conn

    def _querystring(self, limit, **kwargs):
        """ Builds a qury string from kwargs for fetching points """
        qs = {
          "limit": limit,
          "includeLinks": True
        }
        qs.update(**kwargs)
        return urlencode(qs)

    def fetch_index(self, idx, group="train"):
        """ Fetch a single data point at a given index in the dataset """
        qs = urlencode({"limit": 1, "offset": idx, "group": group, "includeLinks": True})
        p = self.conn.get("{}/data/{}/datapoints?{}".format(HOST, self.id, qs)).json()[0]
        return DataPoint(p, shape=self.shape, dtype=self.dtype, mlType=self.mlType)

    def fetch(self, _id):
        """ Fetch a point for a given ID """
        return DataPoint(self.conn.get("{}/datapoints/{}".format(HOST, _id)).json(),
                  shape=self.shape, dtype=self.dtype, mlType=self.mlType)

    def fetch_points(self, limit, **kwargs):
        """ Fetch a list of datapoints """
        qs = self._querystring(limit, **kwargs)
        points = [DataPoint(p, shape=self.shape, dtype=self.dtype, mlType=self.mlType)
                      for p in self.conn.get('{}/data/{}/datapoints?{}'.format(HOST, self.id, qs)).json()]
        return points

    def save(self, auto_cache=True):
        """
          Saves a dataset in the DB. Contains logic for determining whether
          the data should be posted as a single h5 cache or a series of smalled chunked files.
          Upon save completing sets the _index property so that new datapoints are indexed correctly.
        """
        meta = self.meta
        meta.update({
            "shape": list(self.shape),
            "dtype": str(self.image.dtype),
            "sensors": self.sensors,
            "graph": self.image.rda_id,
            "node": self.image.rda.graph()['nodes'][0]['id']
        })

        with open(self.geojson, 'r') as fh:
            mfile = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            body = {
                'metadata': (None, json.dumps(meta), 'application/json'),
                'file': (os.path.basename(self.geojson), mfile, 'application/text')
            }
            doc = self.conn.post(self._data_url, files=body).json()
            self.id = doc["data"]["id"]
            self.links = doc["links"]

            if HOST == "http://localhost:3002":
                links = self.links.copy()
                for key in links:
                    href = links[key]['href']
                    href.replace("host.docker.internal", "localhost")
                    links[key]['href'] = href
                self.links = links

        return doc

    def create(self, data):
        return self.conn.post(self.links["create"]["href"], json=data).json()

    def update(self, data):
        return self.conn.put(self.links["update"]["href"], json=data).json()

    def remove(self):
        self.conn.delete(self.links["delete"]["href"])

    def _publish(self):
        return self.conn.put(self.links["publish"]["href"], json={"public": True}).json()

    def _unpublish(self):
        return self.conn.put(self.links["publish"]["href"], json={"public": False}).json()

    def _release(self, version):
        r = self.conn.post(self.links["release"]["href"], json={"version": version})
        r.raise_for_status()
        return r.json()

    def __del__(self):
        try:
            self._temp_gen.remove()
        except Exception:
            pass


class VedaCollection(BaseSet):
    """
      Creates, persists, and provided access to ML training data via the Sandman Api

      Args:
          name (str): a name for the TrainingSet
      Params:
          mlType (str): the type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'
          bbox (list): a list of geographic coordinate bounds for the data ([minx, miny, maxx, maxy])
          classes (list): a list of names of the classes in the training data (ie buildings, cars, planes, trees)
          source (str): Defaults to 'rda'. Currently anything but RDA imagery is treated as MapsAPI data.
          shape (tuple): the shape of the imagery stored in the data. Used to enforce consistent shapes in the set.
          dtype (str): the dtype of the imergy (ie int8, uint16, float32, etc)
    """
    def __init__(self, geojson, image, name, shape=None, mlType="classification", bbox=None, **kwargs):
        assert mlType in valid_mltypes, "mlType {} not supported. Must be one of {}".format(mlType, valid_mltypes)
        super(VedaCollection, self).__init__()

        self.image = image
        self.geojson = None
        self.bbox = bbox

        if geojson is not None:
            with NamedTemporaryFile(prefix="veda", suffix="json", delete=False) as temp:
                with open(temp.name, 'w') as fh:
                    geojson = json.loads(json.dumps(geojson)) # seriously wtf
                    fh.write(json.dumps(geojson))
                self.geojson = temp.name

        self.id = kwargs.get('id', None)
        self.links = kwargs.get('links')
        if shape is None:
            if self.image is not None:
                self.shape = self.image.chunksize
            else:
                self.shape = (8,256,256)
        else:
            self.shape = tuple(map(int, shape))
        self.dtype = kwargs.get('dtype', None)
        self.percent_cached = kwargs.get('percent_cached', 0)
        self.sensors = kwargs.get('sensors', [image.__class__.__name__])
        self._count = kwargs.get('count', 0)
        self._datapoints = None

        self.meta = {
            "name": name,
            "mlType": mlType,
            "public": kwargs.get("public", False),
            "partition": kwargs.get("partition", [100,0,0]),
            "cache_type": kwargs.get("cache_type", "fetch"),
            "classes": kwargs.get("classes", [])
        }

        for k,v in self.meta.items():
            setattr(self, k, v)

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a TrainingSet """
        doc['data']['links'] = doc['links']
        return cls(None, None, **doc['data'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a TrainingSet """
        url = "{}/data/{}".format(HOST, _id)
        doc = conn.get(url).json()
        return cls.from_doc(doc)

    @property
    def count(self):
        return self._count

    @property
    def type(self):
        return self.meta['mlType']

    def _update_sensors(self, image):
        """ Appends the sensor name to the list of already cached sensors """
        self.sensors.append(image.__class__.__name__)
        self.sensors = list(set(self.sensors))

    def publish(self):
        """ Make a saved TrainingSet publicly searchable and consumable """
        assert self.id is not None, 'You can only publish a saved TrainingSet. Call the save method first.'
        return self._publish()

    def unpublish(self):
        """ Unpublish a saved TraininSet (make it private) """
        assert self.id is not None, 'You can only publish a saved TrainingSet. Call the save method first.'
        return self._unpublish()

    def release(self, version):
        """ Create a released version of this training set. Publishes the entire set to s3."""
        assert self.id is not None, 'You can only release a saved TrainingSet. Call the save method first.'
        return self._release(version)

    def __getitem__(self, slc):
        """ Enable slicing of the TrainingSet by index/slice """
        if slc.__class__.__name__ == 'int':
            start = slc
            limit = 1
        else:
            start, stop = slc.start, slc.stop
            limit = stop - start
        return self.fetch_points(limit, offset=start)