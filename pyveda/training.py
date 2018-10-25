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
from shapely.geometry import shape as shp, mapping
from shapely import ops

from functools import partial
import dask

import threading

from .hdf5 import ImageTrainer
from .rda import MLImage
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
valid_matches = ['INSIDE', 'INTERSECT', 'ALL']

def search(params={}):
    r = conn.post('{}/{}'.format(HOST, "search"), json=params)
    r.raise_for_status()
    #try:
    results = r.json()
    return [VedaCollection.from_doc(s) for s in results]
    #except Exception as err:
    #    print(err)
    #    return []

def vec_to_raster(vectors, shape):
    try:
        arr = rasterize(((g, 1) for g in vectors), out_shape=shape[1:])
    except Exception as err:
        arr = np.zeros(shape[1:])
    return np.array(arr)

class DataPoint(object):
    """ Methods for accessing training data pairs """
    def __init__(self, item, shape=(3,256,256), dtype="uint8", **kwargs):
        self.conn = conn
        self.data = item["data"]
        self.links = item["links"]
        self.imshape = tuple(map(int, shape))
        self.dtype = dtype
        self.ml_type = kwargs.get('mlType')
        self._y = None

        #if self.ml_type is not None and self.ml_type == 'segmentation':
        #    self.data['label'] = vec_to_raster(self.data['label'], self.imshape)


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
        return self.conn.delete(self.links["delete"]["href"]).json()

    def _map_labels(self):
        """ Convert labels to data """
        _type = self.ml_type
        if _type is not None:
            if _type == 'classification':
                return [int(self.label[key]) for key in list(self.label.keys())]
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
        self._bulk_data_url = "{}/data/bulk".format(HOST)
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

    def fetch_index(self, idx):
        """ Fetch a single data point at a given index in the dataset """
        params = {"limit": 1, "offset": idx, "includeLinks": True}
        if self.classes and len(self.classes):
            params["classes"] = ','.join(self.classes)
        qs = urlencode(params)
        p = self.conn.get("{}/data/{}/datapoints?{}".format(HOST, self.id, qs)).json()[0]
        return DataPoint(p, shape=self.imshape, dtype=self.dtype, mlType=self.mlType)

    def fetch(self, _id):
        """ Fetch a point for a given ID """
        qs = urlencode({})
        if self.classes and len(self.classes):
            params = {"classes": ','.join(self.classes)}
            qs = urlencode(params)
        return DataPoint(self.conn.get("{}/datapoints/{}?{}".format(HOST, _id, qs)).json(),
                  shape=self.imshape, dtype=self.dtype, mlType=self.mlType)

    def fetch_points(self, limit, offset=0, **kwargs):
        """ Fetch a list of datapoints """
        params = {"offset": offset}
        if self.classes and len(self.classes):
            params["classes"] = ','.join(self.classes)
        qs = self._querystring(limit, **params)
        points = [DataPoint(p, shape=self.imshape, dtype=self.dtype, mlType=self.mlType)
                      for p in self.conn.get('{}/data/{}/datapoints?{}'.format(HOST, self.id, qs)).json()]
        return points

    def fetch_ids(self, page_size=100, page_id=None):
        """ Fetch a point for a given ID """
        data = self.conn.get("{}/data/{}/ids?pageSize={}&pageId={}".format(HOST, self.id, page_size, page_id)).json()
        return data['ids'], data['nextPageId']

    def _bulk_load(self, s3path, **kwargs):
        meta = self.meta
        meta.update({
            "imshape": list(self.imshape),
            "sensors": self.sensors,
            "dtype": self.dtype
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
            self.id = doc["data"]["id"]
            self._set_links(doc["links"])
        return doc
        

    def _load(self, geojson, image, **kwargs):
        """
            Loads a geojson file into the VC
        """
        meta = self.meta
        meta.update({
            "imshape": list(self.imshape),
            "sensors": self.sensors,
            "dtype": self.dtype
        })
        options = {
            'match':  kwargs.get('match', 'INTERSECTS'),
            'default_label': kwargs.get('default_label', None),
            'label_field':  kwargs.get('label_field', None),
            'cache_type':  kwargs.get('cache_type', 'stream'),
            'graph': image.rda_id,
            'node': image.rda.graph()['nodes'][0]['id'],
            'workers': kwargs.get('workers', 1)
        }
        if 'mask' in kwargs:
            options['mask'] = shape(kwargs.get('mask')).wkt 

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
                self.id = doc["data"]["id"]
                self._set_links(doc["links"])
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
        if HOST == "http://localhost:3002":
            _links = self.links.copy()
            for key in _links:
                href = _links[key]['href']
                href.replace("host.docker.internal", "localhost")
                _links[key]['href'] = href
            self.links = _links


class VedaCollection(BaseSet):
    """
      Creates, persists, and provided access to ML training data via the Sandman Api

      Args:
          name (str): a name for the TrainingSet
      Params:
          mlType (str): the type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'
          tilesize (tuple): the shape of the imagery stored in the data. Used to enforce consistent shapes in the set.
          partition (str):internally partition the contents into `train,validate,test` groups, in percentages. Default is `[100, 0, 0]`, all datapoints in the training group.
    """
    def __init__(self, name, mlType="classification", tilesize=[256,256], partition=[100,0,0], **kwargs):
        assert mlType in valid_mltypes, "mlType {} not supported. Must be one of {}".format(mlType, valid_mltypes)
        super(VedaCollection, self).__init__()
        #default to 0 bands until the first load
        if 'imshape' in kwargs:
            self.imshape = tuple(map(int, kwargs['imshape']))
        else:
            self.imshape = [0] + list(tilesize)
        self.partition = partition
        self.dtype = kwargs.get('dtype', None)
        self.percent_cached = kwargs.get('percent_cached', 0)
        self.sensors = kwargs.get('sensors', [])
        self._count = kwargs.get('count', 0)
        self._datapoints = None
        self.id = kwargs.get('id', None)
        self.links = kwargs.get('links')

        self.meta = {
            "name": name,
            "mlType": mlType,
            "public": kwargs.get("public", False),
            "partition": kwargs.get("partition", [100,0,0]),
            "rda_templates": kwargs.get("rda_templates", []),
            "classes": kwargs.get("classes", []),
            "bbox": kwargs.get("bbox", None),
            "user_id": kwargs.get("userId", None)
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
        
        `cache_type`: The type of caching to use on the server. Valid types are `stream` and `fetch`.

        `workers`: The number of workers on the server to use for caching.'''

        
        assert match.upper() in valid_matches, "match {} not supported. Must be one of {}".format(match, valid_matches)
        # set up the geojson file for upload
        if type(geojson) == str: 
            if not os.path.exists(geojson):
                raise ValueError('{} does not exist'.format(geojson))
        else:
            with NamedTemporaryFile(prefix="veda", suffix="json", delete=False) as temp:
                with open(temp.name, 'w') as fh:
                    geojson = json.loads(json.dumps(geojson)) # seriously wtf
                    fh.write(json.dumps(geojson))
                geojson = temp.name
        if not self.dtype:
            self.dtype = image.dtype.name
        else:
            if self.dtype != image.dtype.name:
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
        doc['data']['links'] = doc['links']
        return cls(**doc['data'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a VedaCollection """
        url = "{}/data/{}".format(HOST, _id)
        r = conn.get(url)
        r.raise_for_status()
        return cls.from_doc(r.json())

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


    @property
    def type(self):
        return self.meta['mlType']

    def ids(self, size=None, page_size=100):
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
                    yield i

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
