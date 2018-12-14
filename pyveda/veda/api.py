import os
import shutil
import json
import math
import time
import sys
import mmap
import inspect
try:
    from urllib.parse import urlencode
except:
    from urllib import urlencode
from collections import namedtuple
from tempfile import NamedTemporaryFile
import numpy as np

from pyveda.auth import Auth
from shapely.geometry import shape as shp, mapping, box

from pyveda.fetch.compat import build_vedabase
from pyveda.vv.labelizer import Labelizer
from pyveda.veda.props import prop_wrap, VEDAPROPS

VALID_MLTYPES = ['classification', 'object_detection', 'segmentation']
VALID_MATCHTYPES = ['INSIDE', 'INTERSECT', 'ALL']

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection




class VedaCollectionBuilder:

    def _load(self, geojson, image, meta=None, **kwargs):
        """
            Loads a geojson file into the VC
        """
        if not meta:
            try:
                meta = self.meta.copy()
                _id = self.id
            except:
                raise AttributeError("Provide metaprop dict and Collection id")
        rda_node = image.rda.graph()['nodes'][0]['id']
        options = {
            'match':  kwargs.get('match', 'INTERSECTS'),
            'default_label': kwargs.get('default_label'),
            'label_field':  kwargs.get('label_field'),
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
            doc = self.conn.post(self._load_url, files=body).json()
        return doc

    def _bulk_load(self, s3path, **kwargs):
        meta = self.meta
        meta.update({
            "imshape": list(self.imshape),
            "sensors": self.sensors,
            "dtype": self.dtype.name
        })
        options = {
            'default_label': kwargs.get('default_label'),
            'label_field':  kwargs.get('label_field'),
            's3path': s3path
        }
        body = {
            'metadata': meta,
            'options': options
        }
        if self.id:
            doc = self.conn.post(self.links["self"]["href"], json=body).json()
        else:
            doc = self.conn.post(self._bulk_data_url, json=body).json()
            self.id = doc["properties"]["id"]
            self._set_links(doc["properties"]["links"])
        return doc



class BaseEndpointConstructor(object):
    """ Builds Veda API endpoints for Collection and Point access methods """

    _veda_load_furl = "{host_url}/data"
    _veda_bulk_load_furl = "{host_url}/data/bulk"
    _dataset_base_furl = "{host_url}/data/{dataset_id}"
    _dataset_create_furl = "{host_url}/datapoints"
    _dataset_release_furl = "/".join([_dataset_base_furl, "release"])
    _dataset_publish_furl = "/".join([_dataset_base_furl, "publish"])
    _datapoint_create_furl = None
    _datapoint_append_furl = None
    _datapoint_update_furl = None
    _datapoint_remove_furl = None

    def __init__(self, host, dataset_id=None):
        self._host = host
        self._dataset_id = dataset_id

    @property
    def _load_url(self):
        return self._veda_load_furl.format(host_url=self._host)

    @property
    def _bulk_load_url(self):
        return self._veda_bulk_load_furl.format(host_url=self._host)

    @property
    def _base_url(self):
        return self._dataset_base_furl.format(host_url=self._host,
                                               dataset_id=self._dataset_id)



_bec = BaseEndpointConstructor

class DataSampleClient(BaseEndpointConstructor):
    """ Veda API wrapper for remote DataSample-relevant methods """
    _accessors = ["create", "append", "update", "remove"]

    def __init__(self, host=None, dataset_id=None):
        self._host = host
        self._dataset_id = None

    @property
    def _create_url(self, dpid):
        raise NotImplementedError

    @property
    def _append_url(self, dpid):
        raise NotImplementedError

    @property
    def _update_url(self, dpid):
        raise NotImplementedError

    @property
    def _remove_url(self, dpid):
        raise NotImplementedError

    def create(self, dpid):
        raise NotImplementedError

    def append(self, dpid):
        raise NotImplementedError

    def update(self, dpid):
        raise NotImplementedError

    def remove(self, dpid):
        raise NotImplementedError


class DataCollectionClient(BaseEndpointConstructor):
    """ Veda API wrapper for remote DataCollection-relevant methods """

    def __init__(self, host, conn=None):
        super(DataCollectionClient, self).__init__(host)
        if not conn:
            self._conn = conn

    @property
    def _host(self):
        return self._host_

    @_host.setter
    def _host(self, host):
        self._host_ = host

    @property
    def _create_url(self):
        return self._dataset_create_furl.format(host_url=self.host)

    @property
    def _update_url(self):
        return self._base_url

    @property
    def _publish_url(self):
        return self._base_url

    @property
    def _release_url(self):
        return self._base_url

    @property
    def _remove_url(self):
        return self._base_url

    def create(self, data):
        r = self.conn.post(self._create_url, json=data)
        r.raise_for_status()
        return r.json()

    def update(self, data):
        r = self.conn.put(self._update_url, json=data)
        r.raise_for_status()
        return r.json()

    def release(self, version):
        r = self.conn.post(self._release_url, json={"version": version})
        r.raise_for_status()
        return r.json()

    def publish(self):
        r = self.conn.put(self._publish_url, json={"public": True})
        r.raise_for_status()

    def unpublish(self):
        r = self.conn.put(self._publish_url, json={"public": False})
        r.raise_for_status()

    def remove(self):
        r = self.conn.delete(self._remove_url)
        r.raise_for_status()

    @classmethod
    def from_links(cls, links, conn=None):
        parts = urlparse(links["self"]["href"])
        host = "{}://{}".format(parts.scheme, parts.netloc)
        dataset_id = parts.path.split("/")[-1] # Use re.match
        return cls(host, dataset_id, conn=conn)


class _VedaCollectionProxy(DataCollectionClient):
    """ Base class for handling all API interactions on sets."""

    def __init__(self, name=None, dataset_id=None, host=HOST, conn=conn, **kwargs):
        self._meta = {k: v for k, v in kwargs if k in self._metaprops}
        self.name = name
        self.id = dataset_id
        super(_VedaCollectionProxy, self).__init__(host, dataset_id)
        self._dp_client(DataPointClient())

    @property
    def id(self):
        try:
            _id = self._meta["dataset_id"]
        except KeyError:
            raise AttributeError("API access requires setting an existing dataset ID")
        return _id

    @id.setter
    def id(self, _id):
        self._meta["dataset_id"] = _id

    @property
    def meta(self):
        return self._meta

    def _to_dp(self, payload, shape=None, dtype=None, mltype=None, **kwargs):
        if not shape:
            shape = self.imshape
        if not dtype:
            dtype = self.dtype
        if not mltype:
            mltype = self.mltype
        return DataPoint(payload, shape=shape, dtype=dtype, mltype=mltype, **kwargs)

    def _querystring(self, params={}, enc_classes=True, **kwargs):
        """ Builds a query string from kwargs for fetching points """
        params.update(**kwargs)
        if enc_classes and self.classes:
            params["classes"] = ",".join(self.classes)
        return urlencode(params)

    def fetch_dp_ids(self, page_size=100, page_id=None):
        """ Fetch a batch of datapoint ids """
        resp = self.conn.get("{}/ids?pageSize={}&pageId={}".format(self._data_url, page_size, page_id))
        resp.raise_for_status()
        data = resp.json()
        return data['ids'], data['nextPageId']

    def fetch_dps_from_slice(self, idx, num_points=1, include_links=True, **kwargs):
        """ Fetch a single data point at a given index in the dataset """
        qs = self._querystring(offset=idx, limit=num_points, includeLinks=include_links)
        resp = self.conn.get(self._datapoint_search_furl.format(self._datapoint_url, qs))
        resp.raise_for_status()
        dps = [self._to_dp(p, **kwargs) for p in resp.json()]

    def fetch_dp_from_id(self, dp_id, include_links=True, **kwargs):
        """ Fetch a point for a given ID """
        qs = self._querystring(includeLinks=include_links)
        resp = self.conn.get(self._datapoint_search_furl.format(self._base_url, dp_id, qs))
        resp.raise_for_status()
        return self._to_dp(resp.json(), **kwargs)

    def fetch_dp_from_idx(self, idx, **kwargs):
        return self.fetch_dps_from_slice(idx, **kwargs).pop()

    def fetch_dps_from_ids(self, dp_ids=[], **kwargs):
        return [self.fetch_dp_from_id(dp_id) for dp_id in dp_ids]

#    This function needs to only reset self._meta to do inplace=refresh
#    def refresh(self):
#        r = self.conn.get(self._base_url))
#        r.raise_for_status()
#        return VedaCollection.from_doc(**r.json())

    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a VedaCollection"""
        if 'id' in doc['properties']:
            doc['properties']['dataset_id'] = doc['properties']['id']
            del doc['properties']['id']
        return cls(**doc['properties'])

    @classmethod
    def from_id(cls, _id, host=HOST):
        """ Helper method that fetches an id into a VedaCollection """
        host = self._host or HOST
        r = self.conn.get("{}/data/{}".format(host, _id))
        r.raise_for_status()
        doc = r.json()
        doc['properties']['host'] = host
        doc['properties']['id'] = _id
        return cls.from_doc(doc)

    _metaprops = VEDAPROPS


VedaCollectionProxy = prop_wrap(_VedaCollectionProxy)
