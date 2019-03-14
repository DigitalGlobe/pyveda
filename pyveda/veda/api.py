import os
import json
import math
import mmap
try:
    from urllib.parse import urlencode
except:
    from urllib import urlencode
import numpy as np
from skimage.io import imsave
from PIL import Image
import requests
from tempfile import NamedTemporaryFile
from shapely.geometry import box

from pyveda.utils import features_to_pixels, url_to_array
from pyveda.veda.props import prop_wrap, VEDAPROPS
from pyveda.veda.loaders import from_geo, from_tarball
from pyveda.config import VedaConfig

from pyveda.vv.labelizer import Labelizer

cfg = VedaConfig()

VALID_MLTYPES = ['classification', 'object_detection', 'segmentation']
VALID_MATCHTYPES = ['INSIDE', 'INTERSECT', 'ALL']


class VedaUploadError(Exception):
    pass

class BaseEndpointConstructor(VedaConfig):
    """ Builds Veda API endpoints for Collection and Point access methods """

    _veda_load_furl = "{host_url}/data"
    _veda_bulk_load_furl = "{host_url}/data/bulk"
    _dataset_base_furl = "{host_url}/data/{dataset_id}"
    _dataset_create_furl = "{host_url}/data/{dataset_id}/datapoints"
    _dataset_release_furl = "/".join([_dataset_base_furl, "release"])
    _dataset_publish_furl = "/".join([_dataset_base_furl, "publish"])
    _datapoint_base_furl = "{host_url}/datapoints/{datapoint_id}"
    _datapoint_search_furl = "{base_url}/datapoints?{qs}"
    _datapoint_fetch_furl = _datapoint_base_furl + "?{qs}"

    @property
    def _load_geo_url(self):
        return self._veda_load_furl.format(host_url=self.host)

    @property
    def _load_tarball_url(self):
        return self._veda_bulk_load_furl.format(host_url=self.host)



class BaseClient(BaseEndpointConstructor):

    @property
    def _data_url(self):
        return self._veda_load_furl.format(host_url=self.host)

    @property
    def _base_url(self):
        return self._dataset_base_furl.format(host_url=self.host,
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
    _accessors = ["update", "remove"]
    def __init__(self, payload, **kwargs):
        self.data = payload['properties']
        self.links = payload["properties"].get("links")
        for k,v in self.data.items():
            setattr(self, k, v)

    @property
    def _url(self):
        return self._datapoint_base_furl.format(host_url=self.host, datapoint_id=self.id)

    @property
    def image(self):
        url = self.links['image']['href']
        return url_to_array(url, self.conn.access_token)

    @property
    def preview(self):
        url = self.links['thumbnail']['href']
        img = Image.open(self.conn.get(url, stream=True).raw)
        return np.array(img)

    def _map_data_props(self):
        """
          Lifts everything in self.data to a property on the class
        """
        for k,v in self.data.items():
            setattr(self, k, v)

    def update(self, new_data):
        """
          Updates metadata props with new values in data
        """
        self.data.update(new_data)
        self._map_data_props()
        r = self.conn.put(self._url, json=new_data)
        r.raise_for_status()
        return r.json()

    def remove(self):
        """
          Removes the sample from Veda
        """
        r = self.conn.delete(self._url)
        r.raise_for_status()

    def __repr__(self):
        data = self.data.copy()
        del data['links']
        return str(data)

    @property
    def __geo_interface__(self):
        return box(*self.bounds).__geo_interface__



class DataCollectionClient(BaseClient):
    """ Veda API wrapper for remote DataCollection-relevant methods """

    @property
    def _create_url(self):
        return self._dataset_create_furl.format(host_url=self.host, dataset_id=self.id)

    @property
    def _update_url(self):
        return self._base_url

    @property
    def _publish_url(self):
        return self._base_url

    @property
    def _release_url(self):
        return self._dataset_release_furl.format(host_url=self.host,
                                                 dataset_id=self.id)

    @property
    def _remove_url(self):
        return self._base_url

    def _add_sample(self, data):
        r = self.conn.post(self._create_url, files=data)
        r.raise_for_status()
        return r.json()

    def update(self, data):
        r = self.conn.put(self._update_url, json=data)
        r.raise_for_status()
        self.refresh()
        return r.json()

    def release(self, version):
        r = self.conn.post(self._release_url, json={"version": version})
        r.raise_for_status()
        doc = r.json()
        return doc['properties']['releases'][version]

    def download_release(self, version, path=None):
        self.refresh()
        assert version in self.releases, 'Release version not found'
        assert self.releases[version]['status'] == 'complete', 'You can only download completed releases.'
        path = path if path is not None else './{}-{}.tar.gz'.format(self.id, version)
        try:
            os.makedirs(os.path.dirname(path))
        except Exception as err:
            pass
        r = self.conn.get(os.path.join(self._release_url, version))
        with open(path, 'wb') as fh:
            fh.write(r.content)
        return path

    def publish(self):
        r = self.conn.put(self._publish_url, json={"public": True})
        r.raise_for_status()

    def unpublish(self):
        r = self.conn.put(self._publish_url, json={"public": False})
        r.raise_for_status()

    def remove(self):
        r = self.conn.delete(self._remove_url)
        r.raise_for_status()

    def refresh(self):
        r = self.conn.get(self._base_url)
        r.raise_for_status()
        meta = {k: v for k, v in r.json()['properties'].items() if k in self._metaprops}
        self._meta.update(meta)

    @classmethod
    def _from_links(cls, links):
        parts = urlparse(links["self"]["href"])
        host = "{}://{}".format(parts.scheme, parts.netloc)
        dataset_id = parts.path.split("/")[-1] # Use re.match
        return cls(dataset_id)


_VedaCollectionProxy = prop_wrap(DataCollectionClient, VEDAPROPS)

class VedaCollectionProxy(_VedaCollectionProxy):
    """ Base class for handling all API interactions on sets."""
    _metaprops = VEDAPROPS
    _data_sample_client = DataSampleClient

    def __init__(self, dataset_id, **kwargs):
        self._meta = {k: v for k, v in kwargs.items() if k in self._metaprops}
        self.id = dataset_id
        self._dataset_id = dataset_id

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
        return self._data_sample_client(payload, shape=shape, dtype=dtype, mltype=mltype, **kwargs)

    def _page_sample_ids(self, page_size=100, page_id=None):
        """ Fetch a batch of datapoint ids """
        resp = self.conn.get("{}/{}/ids?pageSize={}&pageId={}".format(self._data_url, self.id, page_size, page_id))
        resp.raise_for_status()
        data = resp.json()
        return data['ids'], data['nextPageId']

    def fetch_samples_from_slice(self, idx, num_points=1, include_links=True, **kwargs):
        """ Fetch a single data point at a given index in the dataset """
        qs = self._querystring(offset=idx, limit=num_points, includeLinks=include_links)
        resp = self.conn.get(self._datapoint_search_furl.format(base_url=self._base_url, qs=qs))
        resp.raise_for_status()
        dps = [self._to_dp(p, dtype=self.dtype, **kwargs) for p in resp.json()]
        if len(dps) == 1:
            return dps[0]
        return dps

    def fetch_sample_from_id(self, dp_id, include_links=True, **kwargs):
        """ Fetch a point for a given ID """
        qs = self._querystring(includeLinks=include_links)
        resp = self.conn.get(self._datapoint_fetch_furl.format(host_url=self.host,
                                                        datapoint_id=dp_id,
                                                        qs=qs))
        resp.raise_for_status()
        return self._to_dp(resp.json(), dtype=self.dtype, **kwargs)

    def fetch_samples_from_ids(self, dp_ids=[], **kwargs):
        return [self.fetch_sample_from_id(dp_id) for dp_id in dp_ids]

    def fetch_sample_from_index(self, idx, **kwargs):
        return self.fetch_samples_from_slice(idx, **kwargs)

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
        label_url = "{}/datapoints/{}?{}".format(self.host, _id, qs)
        image_url = "{}/datapoints/{}/image.tif".format(self.host, _id)
        #return (_id, [label_url, image_url])
        return (label_url, image_url)

    def append_from_geojson(self, geojson, image, **kwargs):
        """
          Appends new samples to the collection from an image and geojson features.
        """
        if image.dtype is not self.dtype:
            raise ValueError("Image dtype must be {} to match collection".format(self.dtype))
        image = np.squeeze(image)
        if self.status == "BUILDING":
            raise VedaUploadError("Cannot load while server-side caching active")
        params = dict(self.meta, sensors=self.sensors, **kwargs)
        doc = from_geo(geojson, image, **params)

    def append_from_tarball(self, s3path, **kwargs):
        """
          Appends new samples to the collection from an image and geojson features.
        """
        from_tarball(s3path, self.meta, conn=self.conn,
                                      url=self._base_url, **kwargs)

    def add_sample(self, image, label):
        """
          Adds a sample (image/label pair) to the collection

          Args:
            image (rda image): The image to use as the sample image. Must match existing dtype and shapes
            label (dict): Depending on the mltype of the collection is either a dict of ints or a dict of arrays (geojson geometries)
        """
        if image.shape != tuple(self.imshape):
            raise ValueError("Image shape must be {} to match collection".format(self.imshape))
        image = np.squeeze(image)

        # converts labels to pixel coords for obj_detect and segmentation
        for cls, val in label.items():
            label[cls] = features_to_pixels(image, val, self.mltype)

        with NamedTemporaryFile(prefix="veda", suffix='.tif', delete=True) as temp:
            imsave(temp.name, image.read())
            mfile = mmap.mmap(temp.fileno(), 0, access=mmap.ACCESS_READ)
            meta = {
                "bounds": image.bounds,
                "label": label,
                "mltype": self.mltype,
                "dtype": self.dtype.name,
                "dataset_id": self.id,
                "tile_coords": [int(image.bounds[0] * 1000), int(image.bounds[1] * 1000)],
                "cached": True,
                "rda_template": image.rda_id
            }
            payload = {
                "file": (os.path.basename(temp.name), mfile, 'application/text'),
                "metadata": (None, json.dumps(meta), 'application/json'),
            }
            doc = self._add_sample(payload)
            return self._to_dp(doc)


    @classmethod
    def from_doc(cls, doc):
        """ Helper method that converts a db doc to a VedaCollection"""
        if 'id' in doc['properties']:
            doc['properties']['dataset_id'] = doc['properties']['id']
            del doc['properties']['id']
        return cls(**doc['properties'])

    @classmethod
    def from_id(cls, _id):
        """ Helper method that fetches an id into a VedaCollection """
        r = cfg.conn.get("{}/data/{}".format(cls.host, _id))
        r.raise_for_status()
        doc = r.json()
        doc['properties']['host'] = host
        doc['properties']['id'] = _id
        return cls.from_doc(doc)

    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n <= self.count:
            dp = self.fetch_samples_from_slice(self.n, num_points=1)
            self.n += 1
            return dp[0]
        else:
            raise StopIteration

    def __getitem__(self, slc):
        """ Enable slicing of the data by index/slice """
        if slc.__class__.__name__ == 'int':
            start = slc
            limit = 1
        else:
            start, stop = slc.start, slc.stop
            limit = (stop-1) - start
        return self.fetch_samples_from_slice(start, num_points=limit)


    def __repr__(self):
        desc = 'VedaCollectionProxy of {} ({})'.format(
            self.name,
            self.id
        )
        return desc

    def __str__(self):
        desc = '{} ({})\n'.format(
            self.name,
            self.id
        )
        desc +='\t- Bounds: {}\n'.format(
            self.bounds,
        )
        desc +='\t- Count: {} samples, {}% cached\n'.format(
            self.count,
            self.percent_cached
        )
        desc += '\t- Chips: {}x{}, {} bands {}\n'.format(
            self.imshape[1],
            self.imshape[2],
            self.imshape[0],
            self.dtype
        )
        plural = 'es'
        if len(self.classes) == 1:
            plural = ''
        desc += '\t- Labels: {} type, {} class{}'.format(
            self.mltype.replace('_',' '),
            len(self.classes),
            plural
        )
        desc += '\n'
        return desc


    @property
    def __geo_interface__(self):
        return box(*self.bounds).__geo_interface__

    def clean(self, count=None, include_background_tiles=True):
        """
        Page through VedaCollection data and flag bad data.
        Params:
            count (int): the number of tiles to clean
            include_background_tiles (bool): include tiles that do not have labels on them
        Returns:
            l: a Labelizer() object
        """
        classes = self.classes
        mltype = self.mltype
        l = Labelizer(self, mltype, count, classes, include_background_tiles)
        l.clean()
        return l

    def preview(self, count=10, include_background_tiles=True):
        """
        Page through VedaCollection data and flag bad data.
        Params:
            count: the number of tiles to clean
        """
        classes = self.classes
        mltype = self.mltype
        Labelizer(self, mltype, count, classes, include_background_tiles).preview()
