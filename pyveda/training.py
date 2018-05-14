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
import numpy as np
from dask import delayed
import dask.array as da
from gbdxtools import Interface
from tempfile import NamedTemporaryFile
from .loaders import load_image
from .utils import transforms

from functools import partial
import dask
threaded_get = partial(dask.threaded.get, num_workers=64)

gbdx = Interface()
headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.access_token)}

HOST = os.environ.get('SANDMAN_API')
if not HOST:
    HOST = "http://localhost:3002"

def search(params={}):
    r = requests.post('{}/{}'.format(HOST, "search"), json=params, headers=headers)
    return [TrainingSet.from_doc(s) for s in r.json()]

class DataPoint(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), dtype="uint8"):
        self.conn = requests.Session()
        self.conn.headers.update( headers )
        self.data = item["data"]
        self.data['y'] = np.array(self.data['y'])
        self.links = item["links"]
        self.shape = tuple(shape)
        self.dtype = dtype

    @property
    def id(self):
        return self.data["id"]

    @property
    def y(self):
        return np.array(self.data["y"])

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

    def __repr__(self):
        return str(self.data)


class BaseSet(object):
    """ Base class for handling all API interactions on sets."""
    def __init__(self, id=None):
        self.id = id
        self._data_url = "{}/data".format(HOST)
        self._datapoint_url = "{}/datapoints".format(HOST)
        self._chunk_size = os.environ.get('VEDA_CHUNK_SIZE', 5000)
        self.conn = requests.Session()
        self.conn.headers.update( headers )

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
        return DataPoint(p, shape=self.shape, dtype=self.dtype)

    def fetch(self, _id):
        """ Fetch a point for a given ID """
        return DataPoint(self.conn.get("{}/datapoints/{}".format(HOST, _id)).json(), shape=self.shape, dtype=self.dtype)

    def fetch_points(self, limit, **kwargs):
        """ Fetch a list of datapoints """
        qs = self._querystring(limit, **kwargs)
        return [DataPoint(p, shape=self.shape, dtype=self.dtype) for p in self.conn.get('{}/data/{}/datapoints?{}'.format(HOST, self.id, qs)).json()]

    def save(self):
        """ 
          Saves a dataset in the DB. Contains logic for determining whether 
          the data should be posted as a single h5 cache or a series of smalled chunked files.

          Upon save completing sets the _index property so that new datapoints are indexed correctly.
        """
        meta = self.meta
        meta.update({
            "count": dict(self._count),
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "sensors": self.sensors
        })

        total = sum(list(meta['count'].values()))
        if total <= self._chunk_size:
            self.cache.close()
            doc = self._create_set(meta, h5=self.fname)
            self._cache = None
        else:
            doc = self._create_set(meta)
            self._send_chunks(doc)

        self.id = doc["data"]["id"]
        self.links = doc["links"]
        self._cache = None
        self._index = total
        return doc

    def _create_set(self, meta, h5=None):
        """ 
          Creates the set in the API/DB
        """
        if h5 is None:
            # Big dataset, create a doc, then post in chunks
            if self.id is not None:
                doc = self.conn.get(self.links['self']['href'])
            else: 
                doc = self.conn.post(self._data_url, json={"metadata": meta})
        else:
            # Small file send all the data
            with open(h5, 'rb') as f:
                mfile = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                files = {
                    'metadata': (None, json.dumps(meta), 'application/json'),
                    'file': (os.path.basename(h5), mfile, 'application/octet-stream')
                }
                if self.links is not None:
                    url = self.links['self']['href']
                    meta.update({"update": True})
                    files["metadata"] = (None, json.dumps(meta), 'application/json')
                else:
                    url = self._data_url
                doc = self.conn.post(url, files=files)
                mfile.close()
        return doc.json()

    def _send_chunks(self, doc):
        """ Chunk up the HDF5 cache into chunks <= chunk_size and post with doc id """
        session = FuturesSession()
        session.headers.update( headers )
        groups = list(self.cache.keys())
        for group in groups:
            count = self._count[group]
            nchunks = math.ceil(count / self._chunk_size)
            idx = self._index
            for chunk in range(nchunks):
                temp = NamedTemporaryFile(prefix="veda", suffix='h5', delete=False)
                chunk_h5 = h5py.File(temp.name, 'a')
                grp = chunk_h5.create_group(group)
                xgrp = grp.create_group("X")
                ygrp = grp.create_group("Y")
                for i in range(self._chunk_size):
                    try:
                        self.cache.copy('{}/X/{}'.format(group, str(idx)), xgrp)
                        self.cache.copy('{}/Y/{}'.format(group, str(idx)), ygrp)
                    except Exception as err:
                        pass
                    idx += 1

                chunk_h5.close()
                f = open(temp.name, 'rb')
                files = {
                    'file': (os.path.basename(temp.name), f, 'application/octet-stream')
                }
                if self.id is not None:
                    meta = {
                        "count": dict(self._count),
                        "sensors": self.sensors,
                        "update": True
                    }
                    files['metadata'] = (None, json.dumps(meta), 'application/json')
                p = session.post(doc['links']['self']['href'], files=files)
                os.remove(temp.name)

    def create(self, data):
        return self.conn.post(self.links["create"]["href"], json=data).json()

    def update(self, data):
        return self.conn.put(self.links["update"]["href"], json=data).json()

    def remove(self):
        self.conn.delete(self.links["delete"]["href"])




class TrainingSet(BaseSet):
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
    def __init__(self, name, mlType="classification", bbox=[], classes=[], source="rda", **kwargs):
        super(TrainingSet, self).__init__()
        self.source = source
        self.id = kwargs.get('id', None)
        self.links = kwargs.get('links')
        self.shape = kwargs.get('shape', None)
        if self.shape is not None:
            self.shape = tuple(self.shape)
        self.dtype = kwargs.get('dtype', None)
        self.percent_cached = kwargs.get('percent_cached', 0)
        self.sensors = kwargs.get('sensors', [])
        self._count = kwargs.get('count', defaultdict(int))
        self._cache = None
        self._datapoints = None
        self.dasks = defaultdict(dict)
        self._index = sum(list(self._count.values()))

        self.meta = {
            "source": source,
            "name": name,
            "bbox": bbox,
            "classes": classes,
            "nclasses": len(classes),
            "mlType": mlType
        }

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a TrainingSet """
        doc['data']['links'] = doc['links']
        return cls(**doc['data'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a TrainingSet """
        url = "{}/data/{}".format(HOST, _id)
        doc = requests.get(url, headers=headers).json()
        return cls.from_doc(doc)

    @property
    def cache(self):
        """ Looks for an existing cache file and creates it if doesnt exist """
        if self._cache is None:
            temp = NamedTemporaryFile(prefix="veda", suffix='h5', delete=False)
            self.fname = temp.name
            self._cache = h5py.File(temp.name, 'a')
        return self._cache

    @property
    def count(self):
        return self._count

    @property
    def type(self):
        return self.meta['mlType']

    def _create_groups(self, group):
        """ Creates a group in the h5 cache """
        grp = self._cache.create_group(group)
        grp.create_group("X")
        grp.create_group("Y")
        if group not in self._count:
            self._count[group] = 0

    def feed(self, data, group="train", verbose=False):
        """
          Feeds data to a cache from a list of image/label pairs

          Args:
              data (list): list of tuples: [(dask, numpy array)]
          Params:
              group (str): the group to store data into. Defaults to "train"
              verbose (bool): whether or not to print logs while feeding the cache
        """
        if verbose:
            print("caching {} {} records".format(len(data), group))

        if group not in self.cache:
            self._create_groups(group)

        for i in range(len(data)):
            if not i % 100 and verbose:
                print("checkpoint", i)

            dsk, labels = data[i]

            if self.shape is None:
                self.shape = dsk.shape
                self.dtype = dsk.dtype

            self._update_sensors(dsk)

            if dsk.shape == self.shape:
                X = transforms(self.source)(dsk)
                self.add_to_cache(X, labels, group=group)
            elif verbose:
                print('Could not add image to set, shape mismatch {} != {}'.format(dsk.shape, self.shape))

    def add_to_cache(self, X, Y, group="train", idx=None):
        """
          Adds single image and label pair to the cache

          Args:
              X (dask): a image dask
              Y (ndarray): an arrayof label data to add
          Params:
              group (str): the group to append to
              idx (int): an index to use to in the cache (None). If defined overwrites that may exist at that index.
        """
        try:
            y = [str(Y.tolist()).encode('utf8')]
        except:
            y = [str(Y).encode('utf8')]

        if idx is None:
            idx = self.count[group] + self._index
            self.cache[group]["X"].create_dataset(str(idx), data=[x.encode('utf8') for x in X])
            self.cache[group]["Y"].create_dataset(str(idx), data=y)
            self._count[group] += 1
        else:
            del self.cache[group]["X"][str(idx)]
            del self.cache[group]["Y"][str(idx)]
            self.cache[group]["X"].create_dataset(str(idx), data=[x.encode('utf8') for x in X])
            self.cache[group]["Y"].create_dataset(str(idx), data=y)

    def add(self, dsk, labels, group="train"):
        """
          Add a image/lable pair to an existing TrainingSet

          Args:
              dsk: a gbdxtools image dask
              labels: an array or nd-array of label data
        """
        assert self.id is not None, 'Can only call add on existing Sets, try calling `feed`.'
        assert dsk.shape == self.shape, 'Mismatching shapes, cannot add {} to set with shape {}'.format(dsk.shape, self.shape)
        self._update_sensors(dsk)
        X = transforms(self.source)(dsk)
        return self.create({
          'x': x,
          'y': labels.tolist(),
          'group': group,
          'dataset': self.id
        })


    def batch(self, size, group="train", to_cache=False):
        """
          Fetches a batch of randomly sampled pairs from either the set
          Args:
              size (int): the sample size of pairs to fetch
          Params:
              group (str): the group to fetch pairs from
              to_cache (bool): fetch the data directly to a h5 file on disk
        """
        points = self.fetch_points(size, shuffle=True, group=group)
        X = da.stack([p.image for p in points])
        Y = [p.y for p in points]
        if not to_cache:
            return X.compute(get=threaded_get), np.array(Y)
        else:
            # TODO: support saving to hdf5 file and return a class that can read data from it
            pass

    def batch_generator(self, size, group="train"):
        """
          Create a generator of randomly sampled pairs from the set
          Args:
              size (int): the sample size of pairs to fetch
          Params:
              group (str): the group to fetch pairs from
              to_cache (bool): fetch the data directly to a h5 file on disk
        """
        points = self.fetch_points(size, shuffle=True, group=group)
        dsk = da.stack([p.image for p in points])
        for i, p in enumerate(points):
            yield dsk[i].compute(), p.y

    def get_one(self, idx, group="train"):
        """
          Gets a pair of data at a given index from either the h5 cache or API

          Params:
              idx (int): index of the pair to fetch
              group (str): the group to fetch pairs from
        """
        if self.id is None:
            X = self.cache[group]["X"][str(idx)]
            Y = self.cache[group]["Y"][str(idx)]
            data = {
                "x": [x.decode('utf8') for x in X.value.tolist()],
                "y": json.loads([y.decode('utf8') for y in Y.value.tolist()][0]),
                "group": group
            }
            return data
        else:
            return self.fetch_index(idx, group=group)

    def update_index(self, idx, data):
        """
          Update a pair at the given index

          Args:
              idx: the index to override
              data: a dict of data to overwrite in the cache
        """
        if self.id is None:
            self.add_to_cache(data["x"], [tuple(map(float, y)) for y in data["y"]], group=data["group"], idx=idx)
        else:
            pnt = self.fetch_index(idx)
            pnt.update(data, save=True)

    def _update_sensors(self, dsk):
        """ Appends the sensor name to the list of already cached sensors """
        self.sensors.append(dsk.__class__.__name__)
        self.sensors = list(set(self.sensors))

    def publish(self):
        """ Make a saved TrainingSet publicly searchable and consumable """
        assert self.id is not None, 'You can only publish a saved TrainingSet. Call the save method first.'
        return self.update({"public": True})

    def unpublish(self):
        """ Unpublish a saved TraininSet (make it private) """
        assert self.id is not None, 'You can only publish a saved TrainingSet. Call the save method first.'
        return self.update({"public": False})

    def __getitem__(self, slc):
        """ Enable slicing of the TrainingSet by index/slice """
        start, stop = slc.start, slc.stop
        limit = stop - start
        return self.fetch_points(limit, offset=start)
