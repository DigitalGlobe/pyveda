import os
import json
import mmap
import numpy as np
from tempfile import NamedTemporaryFile
from pyveda.auth import Auth
from shapely.geometry import shape

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection

def args_to_meta(name, description, dtype, imshape, 
                 mltype, partition, public, sensors):
    """
      Helper method for just building a dict of meta fields to pass to the API
    """
    return {
      'name': name,
      'description': description,
      'dtype': dtype.name,
      'imshape': imshape,
      'mltype': mltype,
      'public': public,
      'partition': partition,
      'sensors': sensors,
      'classes': [],
      'bounds': []
    }


def from_tarball(s3path, meta, default_label=None,
                                  label_field=None, conn=conn,
                                  url="{}/data/bulk".format(HOST)):
    options = {
        'default_label': default_label,
        'label_field':  label_field,
        's3path': s3path
    }
    body = {
        'metadata': meta,
        'options': options
    }
    doc = conn.post(url, json=body).json()
    return doc


def from_geo(geojson, image, name, tilesize=[256,256], match="INTERSECTS",
                              default_label=None, label_field=None,
                              workers=1, cache_type="stream",
                              dtype=None, description='',
                              mltype="classification", public=False,
                              partition=[100,0,0], mask=None
                              url="{}/data".format(HOST), conn=conn, **kwargs):
    """
        Loads a geojson file into the VC

        Args:
          geojson: geojson feature collection, in the following formats:
              - a path to a geojson file
              - a geojson feature collection in Python dictionary format
          image: Any gbdxtools image object. Veda includes the MLImage type configured with the most commonly used options
                 and only requires a Catalog ID.
          name (str): A name for the TrainingSet.
          mltype (str): The type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'.
          tilesize (list): The shape of the imagery stored in the data. Used to enforce consistent shapes in the set.
          partition (list):Internally partition the contents into `train,validate,test` groups, in percentages. Default is `[100, 0, 0]`, all datapoints in the training group.
          imshape (list): Shape of image data. Multiband should be X,N,M. Single band should be 1,N,M.
          dtype (str): Data type of image data.
          description (str): An optional description of the training dataset. Useful for attaching external info and links to a collection.
          public (bool): Indicates if data is publically available for others to access.
          match: Generates a tile based on the topological relationship of the feature. Can be:
              - `INSIDE`: the feature must be contained inside the tile bounds to generate a tile.
              - `INTERSECTS`: the feature only needs to intersect the tile. The feature will be cropped to the tile boundary (default).
              - `ALL`: Generate all possible tiles that cover the bounding box of the input features, whether or not they contain or intersect features.
          default_label: default label value to apply to all features when  `label` in the geojson `Properties` is missing.
          label_field: Field in the geojson `Properties` to use for the label instead of `label`.
          mask: A geojson geometry to use as a mask with caching tiles. When defined only tile within the mask will be cached.
    """
    if isinstance(geojson, str) and not os.path.exists(geojson):
        raise ValueError('{} does not exist'.format(geojson))
    elif isinstance(geojson, dict):
        with NamedTemporaryFile(mode="w+t", prefix="veda", suffix="json", delete=False) as temp:
            temp.file.write(json.dumps(geojson))
        geojson = temp.name

    assert isinstance(name, str), ValueError('Name must be defined as a string')

    if dtype is None:
       dtype = image.dtype
    else:
       dtype = np.dtype(dtype)

    if dtype.name != image.dtype.name:
       raise ValueError('Image dtype ({}) and given dtype ({}) must match.'.format(image.dtype, dtype))

    sensors = [image.__class__.__name__]
    imshape = [image.shape[0]] + list(tilesize)
    meta = args_to_meta(name, description, dtype, imshape, mltype, partition, public, sensors)

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
    if 'mask' in kwargs:
        options['mask'] = shape(kwargs.get('mask')).wkt

    with open(geojson, 'r') as fh:
        mfile = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        body = {
            'metadata': (None, json.dumps(meta), 'application/json'),
            'file': (os.path.basename(geojson), mfile, 'application/text'),
            'options': (None, json.dumps(options), 'application/json')
        }
        doc = conn.post(url, files=body).json()
    return doc


