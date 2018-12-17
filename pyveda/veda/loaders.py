import os
import json
import mmap
from tempfile import NamedTemporaryFile
from pyveda.auth import Auth

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection


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


def from_geo(geojson, image, meta, match="INTERSECTS",
                              default_label=None, label_field=None,
                              workers=1, cache_type="stream",
                              url="{}/data".format(HOST), conn=conn, **kwargs):
    """
        Loads a geojson file into the VC
    """
    if isinstance(geojson, str) and not os.path.exists(geojson):
        raise ValueError('{} does not exist'.format(geojson))
    elif isinstance(geojson, dict):
        with NamedTemporaryFile(mode="w+t", prefix="veda", suffix="json", delete=False) as temp:
            temp.file.write(json.dumps(geojson))
        geojson = temp.name

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
        options['mask'] = shp(kwargs.get('mask')).wkt

    with open(geojson, 'r') as fh:
        mfile = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        body = {
            'metadata': (None, json.dumps(meta), 'application/json'),
            'file': (os.path.basename(geojson), mfile, 'application/text'),
            'options': (None, json.dumps(options), 'application/json')
        }
        doc = conn.post(url, files=body).json()
    return doc


