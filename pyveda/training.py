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
from gbdxtools import Interface, CatalogImage
from .loaders import load_image
from .utils import transforms
from rasterio.features import rasterize
from shapely.geometry import shape as shp, mapping, box
from shapely import ops

from functools import partial
import dask

import threading

from pyveda.db import VedaBase
from .rda import MLImage
from pyveda.utils import NamedTemporaryHDF5Generator
from pyveda.fetch.compat import build_vedabase

threads = int(os.environ.get('GBDX_THREADS', 64))
threaded_get = partial(dask.threaded.get, num_workers=threads)
gbdx = Interface()

HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")

headers = {"Authorization": "Bearer {}".format(gbdx.gbdx_connection.access_token)}
conn = requests.Session()
conn.headers.update(headers)

valid_mltypes = ['classification', 'object_detection', 'segmentation']
valid_matches = ['INSIDE', 'INTERSECT', 'ALL']


def search(params={}, host=HOST):
    r = conn.post('{}/{}'.format(host, "search"), json=params)
    r.raise_for_status()
    try:
        results = r.json()
        return [VedaCollection.from_doc(s) for s in results]
    except Exception as err:
        print(err)
        return []


class DataPoint(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), **kwargs):
        self.conn = conn
        self.links = item["properties"].get("links")
        self.imshape = tuple(map(int, shape))
        self._y = None

        del item['properties']['links']
        self.data = item['properties']


    @property
    def id(self):
        return self.data["id"]

    @property
    def mltype(self):
        return self.data.get('mlType')

    @property
    def dtype(self):
        return np.dtype(self.data.get('dtype', 'uint8'))

    @property
    def label(self):
        return self.data['label']

    @property
    def bounds(self):
        if 'bounds' in self.data:
            return self.data['bounds']
        else:
            return None

    @property
    def tile_coords(self):
        if 'tile_coords' in self.data:
            return self.data['tile_coords']
        else:
            return None

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
        _type = self.mltype
        if _type is not None:
            if _type == 'classification':
                return [int(self.label[key]) for key in list(self.label.keys())]
            else:
                return self.label
        else:
            return None

    def __repr__(self):
        return str(self.data)

    @property
    def meta(self):
        ''' returns metadata useful to end users '''
        data = self.data.copy()
        parent = self.data['dataset_id']
        try:
            del data['dataset_id']
            del data['queue']
            del data['sha']
            del data['tile_coords']
        except KeyError:
            pass
        data['parent'] = parent
        return data

    @property
    def __geo_interface__(self):
        return box(*self.bounds).__geo_interface__


class BaseSet(object):
    """ Base class for handling all API interactions on sets."""
    def __init__(self, id=None, host=HOST):
        self.id = id
        self._host = host
        self._data_url = "{}/data".format(self._host)
        self._bulk_data_url = "{}/data/bulk".format(self._host)
        self._cache_url = "{}/data/{}/cache"
        self._datapoint_url = "{}/datapoints".format(self._host)
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

    def fetch_index(self, idx):
        """ Fetch a single data point at a given index in the dataset """
        params = {"limit": 1, "offset": idx, "includeLinks": True}
        if self.classes and len(self.classes):
            params["classes"] = ','.join(self.classes)
        qs = urlencode(params)
        p = self.conn.get("{}/data/{}/datapoints?{}".format(self._host, self.id, qs)).json()[0]
        return DataPoint(p, shape=self.imshape, dtype=self.dtype, mlType=self.mltype)

    def fetch(self, _id, **kwargs):
        """ Fetch a point for a given ID """
        params = {"includeLinks": True}
        if self.classes and len(self.classes):
            params.update({"classes": ','.join(self.classes)})
            qs = urlencode(params)
        return DataPoint(self.conn.get("{}/datapoints/{}?{}".format(self._host, _id, qs)).json(),
                  shape=self.imshape, dtype=self.dtype, mlType=self.mltype)

    def fetch_points(self, limit, offset=0, **kwargs):
        """ Fetch a list of datapoints """
        params = {"offset": offset, "includeLinks": True}
        if self.classes and len(self.classes):
            params["classes"] = ','.join(self.classes)
        qs = self._querystring(limit, **params)
        points = [DataPoint(p, shape=self.imshape, dtype=self.dtype, mlType=self.mltype)
                      for p in self.conn.get('{}/data/{}/datapoints?{}'.format(self._host, self.id, qs)).json()]
        return points

    def fetch_ids(self, page_size=100, page_id=None):
        """ Fetch a point for a given ID """
        data = self.conn.get("{}/data/{}/ids?pageSize={}&pageId={}".format(self._host, self.id, page_size, page_id)).json()
        return data['ids'], data['nextPageId']

    def _bulk_load(self, s3path, **kwargs):
        meta = self.meta
        meta.update({
            "imshape": list(self.imshape),
            "sensors": self.sensors,
            "dtype": self.dtype.name
        })
        options = {
            'default_label': kwargs.get('default_label', None),
            'label_field':  kwargs.get('label_field', None),
            's3path': s3path
        }
        body = {
            'metadata': meta,
            'options': options
        }
        if self.id is not None:
            doc = self.conn.post(self.links["self"]["href"], json=body).json()
        else:
            doc = self.conn.post(self._bulk_data_url, json=body).json()
            self.id = doc["properties"]["id"]
            self._set_links(doc["properties"]["links"])
        return doc

    def _load(self, geojson, image, **kwargs):
        """
            Loads a geojson file into the VC
        """
        rda_node = image.rda.graph()['nodes'][0]['id']
        meta = self.meta
        meta.update({
            "imshape": list(self.imshape),
            "sensors": self.sensors,
            "dtype": self.dtype.name
        })
        options = {
            'match':  kwargs.get('match', 'INTERSECTS'),
            'default_label': kwargs.get('default_label', None),
            'label_field':  kwargs.get('label_field', None),
            'cache_type':  kwargs.get('cache_type', 'stream'),
            'graph': image.rda_id,
            'node': rda_node,
            'workers': kwargs.get('workers', 1)
        }
        if 'mask' in kwargs:
            options['mask'] = shp(kwargs.get('mask')).wkt

        with open(geojson, 'r') as fh:
            mfile = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
            body = {
                'metadata': (None, json.dumps(meta), 'application/json'),
                'file': (os.path.basename(geojson), mfile, 'application/text'),
                'options': (None, json.dumps(options), 'application/json')
            }
            if self.id is not None:
                doc = self.conn.post(self.links["self"]["href"], files=body).json()
            else:
                doc = self.conn.post(self._data_url, files=body).json()
                self.id = doc["properties"]["id"]
                self._set_links(doc["properties"]["links"])
        return doc

    def create(self, data):
        return self.conn.post(self.links["create"]["href"], json=data).json()

    def update(self, data):
        self.conn.put(self.links["update"]["href"], json=data).json()

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

    def _refresh(self):
        r = self.conn.get(self.links["self"]["href"])
        r.raise_for_status()
        return VedaCollection.from_doc(**r.json())

    def _set_links(self, links):
        self.links = links
        if self._host == "http://localhost:3002":
            _links = self.links.copy()
            for key in _links:
                href = _links[key]['href']
                href.replace("host.docker.internal", "localhost")
                _links[key]['href'] = href
            self.links = _links


class VedaCollection(BaseSet):
    """
      Creates, persists, and provided access to ML training data via the Veda Api

      Args:
          name (str): a name for the TrainingSet
      Params:
          mlType (str): the type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'
          tilesize (tuple): the shape of the imagery stored in the data. Used to enforce consistent shapes in the set.
          partition (str):internally partition the contents into `train,validate,test` groups, in percentages. Default is `[100, 0, 0]`, all datapoints in the training group.
    """
    def __init__(self, name, mlType="classification", tilesize=[256,256], partition=[100,0,0],
                imshape=None, dtype=None, percent_cached=0, sensors=[], _count=0,
                dataset_id=None, image_refs=None,classes=[], bounds=None,
                user_id=None, public=False, host=HOST, links=None):

        assert mlType in valid_mltypes, "mlType {} not supported. Must be one of {}".format(mlType, valid_mltypes)
        super(VedaCollection, self).__init__()
        #default to 0 bands until the first load
        if 'imshape':
            self.imshape = tuple(map(int, imshape))
        else:
            self.imshape = [0] + list(tilesize)
        self.partition = partition
        self.dtype = dtype
        if self.dtype is not None:
            self.dtype = np.dtype(self.dtype)
        self.percent_cached = percent_cached
        self.sensors = sensors
        self._count = _count
        self._datapoints = None
        self.id = dataset_id
        self.links = kwargs.get('links')
        self._host = kwargs.get('host', HOST)

        self.meta = {
            "name": name,
            "mltype": mlType,
            "public": public,
            "partition": partition,
            "image_refs": image_refs,
            "classes": classes,
            "bounds": bounds,
            "user_id": user_id
        }

        for k,v in self.meta.items():
            setattr(self, k, v)

    def bulk_load(self, s3path, **kwargs):
        self._bulk_load(s3path, **kwargs)


    def load(self, geojson, image, match="INTERSECT", default_label=None, label_field = None, cache_type="stream", **kwargs):
        '''Loads a geojson file or object into the VedaCollection
        ARGS

        `geojson`: geojson feature collection, in the following formats:

        - a path to a geojson file
        - a geojson feature collection in Python dictionary format
        - TODO: a Python list of either of the above
        - TODO: a generator function that emits feature objects in Python dictionary format

        `image`: Any gbdxtools image object. Veda includes the MLImage type configured with the most commonly used options
                 and only requires a Catalog ID.

        `match`: Generates a tile based on the topological relationship of the feature. Can be:

        - `INSIDE`: the feature must be contained inside the tile bounds to generate a tile.
        - `INTERSECTS`: the feature only needs to intersect the tile. The feature will be cropped to the tile boundary (default).
        - `ALL`: Generate all possible tiles that cover the bounding box of the input features, whether or not they contain or intersect features.

        `default_label`: default label value to apply to all features when  `label` in the geojson `Properties` is missing.

        `label_field`: Field in the geojson `Properties` to use for the label instead of `label`.

        `cache_type`: The type of caching to use on the server. Valid types are `stream` and `fetch`.'''


        assert match.upper() in valid_matches, "match {} not supported. Must be one of {}".format(match, valid_matches)
        # set up the geojson file for upload
        if type(geojson) == str:
            if not os.path.exists(geojson):
                raise ValueError('{} does not exist'.format(geojson))
        else:
            with NamedTemporaryFile(mode="w+t", prefix="veda", suffix="json", delete=False) as temp:
                temp.file.write(json.dumps(geojson))
            geojson = temp.name
        if self.dtype is None:
            self.dtype = image.dtype
        else:
            if self.dtype.name != image.dtype.name:
                raise ValueError('Image dtype must be {} to match previous images'.format(self.dtype))
        # set the image bands
        # imshape is N,M for single band; X,N,M for multiband
        if self.imshape[0] == 0:
            if image.shape[0] > 1:
                self.imshape[0] = image.shape[0]
            else:
                self.imshape = self.imshape[1:]
        else:
            # multiband X,N,M
            if len(self.imshape) > 2:
                if self.imshape[0] != image.shape[0]:
                    raise ValueError('Image must have {} bands to match previously loaded images'.format(self.imshape[0]))
            # single band N,M, incoming must be 1,N,M
            else:
                if image.shape[0] != 1:
                    raise ValueError('Image must be single band to match previously loaded images')
        self._update_sensors(image)
        self._load(geojson, image, match=match, default_label=default_label, label_field=label_field, cache_type=cache_type, **kwargs)

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a VedaCollection"""
        return cls(**doc['properties'])

    @classmethod
    def from_id(cls, _id, host=HOST):
        """ Helper method that fetches an id into a VedaCollection """
        url = "{}/data/{}".format(host, _id)
        r = conn.get(url)
        r.raise_for_status()
        doc = r.json()
        doc['properties']['host'] = host
        return cls.from_doc(doc)

    @property
    def count(self):
        return self._count

    @property
    def status(self):
        # update percent_cached?
        if self.percent_cached == None:
            return {'status':'EMPTY'}
        elif self.percent_cached == 100:
            return {'status':'COMPLETE'}
        else:
            return {'status':'BUILDING'}

    def ids(self, size=None, page_size=100, get_urls=True):
        if size is None:
            size = self.count
        def get(pages):
            next_page = None
            for p in range(0, pages):
                ids, next_page = self.fetch_ids(page_size, page_id=next_page)
                yield ids, next_page

        count = 0
        for ids, next_page in get(math.ceil(size/page_size)):
            for i in ids:
                count += 1
                if count <= size:
                    if not get_urls:
                        yield i
                    else:
                        yield self._urls_from_id(i)

    def _urls_from_id(self, _id):
        qs = urlencode({})
        if self.classes and len(self.classes):
            params = {"classes": ','.join(self.classes)}
            qs = urlencode(params)
        label_url = "{}/datapoints/{}?{}".format(self._host, _id, qs)
        image_url = "{}/datapoints/{}/image.tif".format(self._host, _id)
        return [label_url, image_url]

    def _update_sensors(self, image):
        """ Appends the sensor name to the list of already cached sensors """
        self.sensors.append(image.__class__.__name__)
        self.sensors = list(set(self.sensors))

    def publish(self):
        """ Make a saved VedaCollection publicly searchable and consumable """
        assert self.id is not None, 'You can only publish a loaded VedaCollection. Call the load method first.'
        return self._publish()

    def unpublish(self):
        """ Unpublish a saved VedaCollection (make it private) """
        assert self.id is not None, 'You can only publish a loaded VedaCollection. Call the load method first.'
        return self._unpublish()

    def release(self, version):
        """ Create a released version of this VedaCollection. Publishes the entire set to s3."""
        assert self.id is not None, 'You can only release a loaded VedaCollection. Call the load method first.'
        return self._release(version)

    def store(self, fname, size=None, partition=[70, 20, 10], **kwargs):
        """ Build an hdf5 database from this collection and return it as a DataSet instance """
        namepath, ext = os.path.splitext(fname)
        if ext != ".h5":
            fname = namepath + ".h5"

        if size is None:
            size = self.count

        pgen = self.ids(size=size)
        vb = VedaBase(fname, self.mltype, self.meta['classes'], self.imshape, image_dtype=self.dtype, **kwargs)

        build_vedabase(vb, pgen, partition, size, gbdx.gbdx_connection.access_token, label_threads=1, image_threads=10)
        vb.flush()
        return vb

    def refresh(self):
        """ Create a released version of this VedaCollection. Publishes the entire set to s3."""
        assert self.id is not None, 'You can only refresh a VedaCollection that has been loaed. Call the load method first.'
        return self._refresh()

    def __getitem__(self, slc):
        """ Enable slicing of the VedaCollection by index/slice """
        if slc.__class__.__name__ == 'int':
            start = slc
            limit = 1
        else:
            start, stop = slc.start, slc.stop
            limit = (stop-1) - start
        return self.fetch_points(limit, offset=start)

    @property
    def __geo_interface__(self):
        return box(*self.bounds).__geo_interface__
