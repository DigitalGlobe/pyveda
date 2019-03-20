import os
import json
import mmap
import numpy as np
import requests
from tempfile import NamedTemporaryFile
from shapely.geometry import shape
import warnings
from pyveda.config import VedaConfig

cfg = VedaConfig()

def args_to_meta(name, description, dtype, imshape,
                 mltype, public, sensors, background_ratio=None):
    """
      Helper method for just building a dict of meta fields to pass to the API
    """
    meta_dict = {
      'name': name,
      'description': description,
      'dtype': dtype.name,
      'imshape': imshape,
      'mltype': mltype,
      'public': public,
      'sensors': sensors,
      'classes': [],
      'bounds': None
    }
    if background_ratio is not None:
        meta_dict['background_ratio'] = max(0.0, min(1.0, float(background_ratio)))
    return meta_dict


def from_tarball(s3path, name=None, dtype='uint8',
                                    imshape=[3,256,256],
                                    label_field=None,
                                    default_label=None,
                                    mltype="classification",
                                    description="",
                                    public=False,
                                    sensors=[]):
    dtype = np.dtype(dtype)
    meta = args_to_meta(name, description, dtype, imshape, mltype, public, sensors)
    options = {
        'default_label': default_label,
        'label_field':  label_field,
        's3path': s3path
    }
    body = {
        'metadata': meta,
        'options': options
    }
    url = "{}/data/bulk".format(cfg.host)
    doc = cfg.conn.post(url, json=body).json()
    return doc


def from_geo(geojson, image, name=None, tilesize=[256,256], match="INTERSECT",
                              default_label=None, label_field=None,
                              workers=1, cache_type="stream",
                              dtype=None, description='',
                              mltype="classification", public=False,
                              sensors=[],
                              background_ratio=0.0,
                              **kwargs):
    """
        Loads a geojson file into the collection

        Args:
          geojson: geojson feature collection, in the following formats:
              - a path to a geojson file
              - a geojson feature collection in Python dictionary format
          image: Any gbdxtools image object. Veda includes the MLImage type configured with the most commonly used options
                 and only requires a Catalog ID.
          name (str): A name for the TrainingSet.
          mltype (str): The type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'.
          tilesize (list): The shape of the imagery stored in the data. Used to enforce consistent shapes in the set.
          imshape (list): Shape of image data. Multiband should be X,N,M. Single band should be 1,N,M.
          dtype (str): Data type of image data.
          description (str): An optional description of the training dataset. Useful for attaching external info and links to a collection.
          public (bool): Indicates if data is publically available for others to access.
          sensors(lst): The different satellites/sensors used for image sources in this VedaCollection.
          match: Generates a tile based on the topological relationship of the feature. Can be:
              - `INSIDE`: the feature must be contained inside the tile bounds to generate a tile.
              - `INTERSECT`: the feature only needs to intersect the tile. The feature will be cropped to the tile boundary (default).
              - `ALL`: Generate all possible tiles that cover the bounding box of the input features, whether or not they contain or intersect features.
          default_label: default label value to apply to all features when  `label` in the geojson `Properties` is missing.
          label_field: Field in the geojson `Properties` to use for the label instead of `label`.
          mask: A geojson geometry to use as a mask with caching tiles. When defined only tile within the mask will be cached.
    """
    if isinstance(geojson, str) and not os.path.exists(geojson):
        raise ValueError('{} does not exist'.format(geojson))
    elif isinstance(geojson, dict):
        assert len(geojson['features']), "No features found in geojson. At least one feature is needed for creating data."
        with NamedTemporaryFile(mode="w+t", prefix="veda", suffix="json", delete=False) as temp:
            temp.file.write(json.dumps(geojson))
        geojson = temp.name


    if dtype is None:
       dtype = image.dtype
    else:
       dtype = np.dtype(dtype)

    if dtype.name != image.dtype.name:
        warnings.warn('Image dtype ({}) and given dtype ({}) do not match.'.format(image.dtype, dtype))
    #   raise ValueError('Image dtype ({}) and given dtype ({}) must match.'.format(image.dtype, dtype))

    imshape = [image.shape[0]] + list(tilesize)
    meta = args_to_meta(name, description, dtype, imshape, mltype, public, sensors, background_ratio)

    rda_node = image.rda.graph()['nodes'][0]['id']
    options = {
        'match':  match,
        'default_label': default_label,
        'label_field':  label_field,
        'cache_type':  cache_type,
        'workers': workers,
        'graph': image.rda_id,
        'node': rda_node,
    }
    if 'mask' in kwargs and kwargs.get('mask'):
        options['mask'] = shape(kwargs.get('mask')).wkt

    with open(geojson, 'r') as fh:
        mfile = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        body = {
            'metadata': (None, json.dumps(meta), 'application/json'),
            'file': (os.path.basename(geojson), mfile, 'application/text'),
            'options': (None, json.dumps(options), 'application/json')
        }
        url = "{}/data".format(cfg.host)
        if 'dataset_id' in kwargs:
            url += "/{}".format(kwargs['dataset_id'])
        r = cfg.conn.post(url, files=body)
        if r.status_code <= 201:
            return r.json()
        else:
            raise requests.exceptions.HTTPError(r.json())
