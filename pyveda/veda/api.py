import os
import json
import math
from functools import partial
try:
    from urllib.parse import urlencode
except:
    from urllib import urlencode
import numpy as np

from pyveda.auth import Auth
from shapely.geometry import shape as shp, mapping, box

from pyveda.veda.props import prop_wrap, VEDAPROPS
from pyveda.veda.loaders import from_geo, from_tarball

VALID_MLTYPES = ['classification', 'object_detection', 'segmentation']
VALID_MATCHTYPES = ['INSIDE', 'INTERSECT', 'ALL']

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection



class VedaUploadError(Exception):
    pass

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

    def __init__(self, host):
        self._host_ = host

    @property
    def _host(self):
        return self._host_

    @_host.setter
    def _host(self, host):
        self._host_ = host

    @property
    def _load_geo_url(self):
        return self._veda_load_furl.format(host_url=self._host)

    @property
    def _load_tarball_url(self):
        return self._veda_bulk_load_furl.format(host_url=self._host)



class BaseClient(BaseEndpointConstructor):

    @property
    def _data_url(self):
        return self._veda_load_furl.format(host_url=self._host)

    @property
    def _base_url(self):
        return self._dataset_base_furl.format(host_url=self._host,
                                               dataset_id=self._dataset_id)

    def _querystring(self, params={}, enc_classes=True, **kwargs):
        """ Builds a query string from kwargs for fetching points """
        params.update(**kwargs)
        if enc_classes and self.classes:
            params["classes"] = ",".join(self.classes)
        return urlencode(params)



_bec = BaseEndpointConstructor

class DataSampleClient(BaseClient):
    """ Veda API wrapper for remote DataSample-relevant methods """
    _accessors = ["create", "append", "update", "remove"]

    def __init__(self, host=None, dataset_id=None, datasample_id=None):
        self._dataset_id = dataset_id
        self._datasample_id = datasample_id
        super(DataSampleClient, self).__init__(host=host)

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


class DataCollectionClient(BaseClient):
    """ Veda API wrapper for remote DataCollection-relevant methods """

    def __init__(self, host, conn):
        super(DataCollectionClient, self).__init__(host)
        self.conn = conn

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
    def _from_links(cls, links, conn=None):
        parts = urlparse(links["self"]["href"])
        host = "{}://{}".format(parts.scheme, parts.netloc)
        dataset_id = parts.path.split("/")[-1] # Use re.match
        return cls(host, dataset_id, conn=conn)



_VedaCollectionProxy = prop_wrap(DataCollectionClient, VEDAPROPS)

class VedaCollectionProxy(_VedaCollectionProxy):
    """ Base class for handling all API interactions on sets."""
    _metaprops = VEDAPROPS
    _data_sample_client = DataSampleClient

    def __init__(self, dataset_id, host=HOST, conn=conn, **kwargs):
        self._meta = {k: v for k, v in kwargs.items() if k in self._metaprops}
        self.id = dataset_id
        self._dataset_id = dataset_id
        super(VedaCollectionProxy, self).__init__(host, conn)

    @property
    def id(self):
        return self._meta["dataset_id"]

    @id.setter
    def id(self, _id):
        self._meta["dataset_id"] = _id

    @property
    def meta(self):
        return self._meta

    @property
    def status(self):
        self.refresh()
        if self.percent_cached == None:
            return {'status':'EMPTY'}
        elif self.percent_cached == 100:
            return {'status':'COMPLETE'}
        else:
            return {'status':'BUILDING'}

    def _to_dp(self, payload, shape=None, dtype=None, mltype=None, **kwargs):
        if not shape:
            shape = self.imshape
        if not dtype:
            dtype = self.dtype
        if not mltype:
            mltype = self.mltype
        return DataPoint(payload, shape=shape, dtype=dtype, mltype=mltype, **kwargs)

    def _page_sample_ids(self, page_size=100, page_id=None):
        """ Fetch a batch of datapoint ids """
        resp = self.conn.get("{}/{}/ids?pageSize={}&pageId={}".format(self._data_url, self.id, page_size, page_id))
        resp.raise_for_status()
        data = resp.json()
        return data['ids'], data['nextPageId']

    def fetch_samples_from_slice(self, idx, num_points=1, include_links=True, **kwargs):
        """ Fetch a single data point at a given index in the dataset """
        qs = self._querystring(offset=idx, limit=num_points, includeLinks=include_links)
        resp = self.conn.get(self._datapoint_search_furl.format(self._datapoint_url, qs))
        resp.raise_for_status()
        dps = [self._to_dp(p, **kwargs) for p in resp.json()]

    def fetch_sample_from_id(self, dp_id, include_links=True, **kwargs):
        """ Fetch a point for a given ID """
        qs = self._querystring(includeLinks=include_links)
        resp = self.conn.get(self._datapoint_search_furl.format(self._base_url, dp_id, qs))
        resp.raise_for_status()
        return self._to_dp(resp.json(), **kwargs)

    def fetch_sample_from_id(self, idx, **kwargs):
        return self.fetch_dps_from_slice(idx, **kwargs).pop()

    def fetch_samples_from_ids(self, dp_ids=[], **kwargs):
        return [self.fetch_sample_from_id(dp_id) for dp_id in dp_ids]

    def refresh(self):
        r = self.conn.get(self._base_url)
        r.raise_for_status()
        meta = {k: v for k, v in r.json()['properties'].items() if k in self._metaprops}
        self._meta.update(meta)

    def gen_sample_ids(self, count=None, page_size=100, get_urls=True, **kwargs):
        """ Creates a generator of Datapoint IDs or URLs for every datapoint in the VedaCollection
            This is useful for gaining access to the ID or the URL for datapoints.
            Args:
            `count` (int): the total number of points to fetch, defaults to None
            `page_size` (int): the size of the pages to use in the API.
            `get_urls` (bool): generate urls tuples ((`dp_url`, `dp_image_url`)) instead of IDs.
            Returns:
              generator of IDs
        """
        if count is None:
            count = self.count
        if count > self.count:
            raise ValueError("Things not big enough for that")

        def get(pages):
            next_page = None
            for p in range(0, pages):
                ids, next_page = self._page_sample_ids(page_size, page_id=next_page)
                yield ids, next_page

        _count = 0
        for ids, next_page in get(math.ceil(count/page_size)):
            for i in ids:
                _count += 1
                if _count <= count:
                    if not get_urls:
                        yield i
                    else:
                        yield self._sample_urls_from_id(i)

    def _sample_urls_from_id(self, _id):
        qs = self._querystring(includeLinks=False)
        label_url = "{}/datapoints/{}?{}".format(self._host, _id, qs)
        image_url = "{}/datapoints/{}/image.tif".format(self._host, _id)
        #return (_id, [label_url, image_url])
        return (label_url, image_url)

    def append_from_geojson(self, geojson, image, **kwargs):
        if image.dtype is not self.dtype:
            raise ValueError("Image dtype must be {} to match collection".format(self.dtype))
        image = np.squeeze(image)
        #if image.shape != self.imshape:
        #    raise ValueError("Image shape must be {} to match collection".format(self.imshape))
        #self.refresh()
        if self.status == "BUILDING":
            raise VedaUploadError("Cannot load while server-side caching active")
        self.sensors.append(image.__class__.__name__)
        params = dict(self.meta, url=self._base_url, conn=self.conn, **kwargs)
        doc = from_geo(geojson, image, **params)

    def append_from_tarball(self, s3path, **kwargs):
        from_tarball(s3path, self.meta, conn=self.conn,
                                      url=self._base_url, **kwargs)

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
        r = conn.get("{}/data/{}".format(host, _id))
        r.raise_for_status()
        doc = r.json()
        doc['properties']['host'] = host
        doc['properties']['id'] = _id
        return cls.from_doc(doc)



