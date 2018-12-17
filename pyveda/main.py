import os
from pyveda.exceptions import RemoteCollectionNotFound
from pyveda.auth import Auth
from pyveda.vedaset import VedaBase, VedaStream
from pyveda.veda.loaders import from_geo, from_tarball
from pyveda.veda.api import _bec, VedaCollectionProxy
from pyveda.veda.rda import MLImage
gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection

__all__ = ["search",
           "load",
           "store",
           "dataset_exists",
           "create_from_geojson",
           "create_from_tarball"]

def _map_contains_submap(mmap, submap, hard_match=True):
    """
    Checks if a submap(submap)(dict) is contained in a master map(mmap)((dict).
    If hard_match is True (default), a submap containing
    keys not defined in the master map will return False. Otherwise,
    returns True if all submap keys map to values equal to master mapping.
    Map values must define __eq__.
    """
    mset, sset = set(mmap.keys()), set(submap.keys())
    if hard_match and not mset.issuperset(sset):
        return False
    shared = mset.intersection(sset)
    if not shared:
        return True
    return all([mmap[key] == submap[key] for key in shared])


def search(params={}, host=HOST, filters={}, **kwargs):
    r = conn.post('{}/{}'.format(host, "search"), json=params)
    r.raise_for_status()
    results = r.json()
    return [VedaCollectionProxy.from_doc(s) for s in results
            if _map_contains_submap(s["properties"], filters, **kwargs)]


def dataset_exists(dataset_id=None, dataset_name=None, conn=conn, host=HOST,
                   return_coll=True):
    if dataset_id:
        r = conn.get(_bec._dataset_base_furl.format(host_url=host,
                                                    dataset_id=dataset_id))
        r.raise_for_status()
        if r.status_code == 200:
            return True if not return_coll else VedaCollectionProxy.from_doc(r.json())
    if dataset_name:
        results = search(filters={"name": dataset_name})
        if results:
            return True if not return_coll else results[0]
        return False

    raise ValueError("Must provide dataset_id or name arguments")


def load(dataset_id=None, dataset_name=None, filename=None, count=None,
         partition=[70,20,10], **kwargs):
    """
    Main interface to access to remote, local and synced datasets
    """

    if not dataset_id or dataset_name or filename:
        raise ValueError("When calling pyveda.load, specify one of: dataset_id, dataset_name, or filename")
    # Check for dataset on veda
    vcp = False
    if dataset_id:
        vcp = dataset_exists(dataset_id=dataset_id)
    elif dataset_name:
        vcp = dataset_exists(dataset_name=dataset_name)
    if vcp:
        return _load_existing(vcp, partition=partition, **kwargs)
    if filename:
        return _load_store(filename, **kwargs)
    raise RemoteCollectionNotFound("No Collection found on Veda for identifier: {}".format(identifier))


def store(filename, dataset_id=None, dataset_name=None, count=None,
          partition=[70,20,10], **kwargs):
    """
    Download a collection locally into a VedaBase hdf5 store
    """
    if not dataset_id or dataset_name:
        raise ValueError("When calling pyveda.store, specify one of: dataset_id, dataset_name")
    coll = dataset_exists(dataset_id=dataset_id, dataset_name=dataset_name)
    vb = VedaBase._from_vedacollection(coll, count=count, partition=partition, **kwargs)
    urlgen = vb._generate_sample_urls()
    build_vedabase(vb, gbdx.gbdx_connection.access_token, **kwargs)
    vb.flush()
    return vb


def _load_existing(vc, *args, **kwargs):
    return VedaStream.from_vc(vc, *args, **kwargs)


def _load_store(filename, **kwargs):
    return VedaBase.from_path(filename, **kwargs)

def create_from_geojson(geojson, image, name, tilesize=[256,256], match="INTERSECT",
                              default_label=None, label_field=None,
                              workers=1, cache_type="stream",
                              dtype=None, description='',
                              mltype="classification", public=False,
                              partition=[100,0,0], mask=None,
                              url="{}/data".format(HOST), conn=conn, **kwargs):
    """
        Loads geojson and an image into a new collection of data

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
              - `INTERSECT`: the feature only needs to intersect the tile. The feature will be cropped to the tile boundary (default).
              - `ALL`: Generate all possible tiles that cover the bounding box of the input features, whether or not they contain or intersect features.
          default_label: default label value to apply to all features when  `label` in the geojson `Properties` is missing.
          label_field: Field in the geojson `Properties` to use for the label instead of `label`.
          mask: A geojson geometry to use as a mask with caching tiles. When defined only tile within the mask will be cached.
    """
    assert isinstance(name, str), ValueError('Name must be defined as a string')

    sensors = [image.__class__.__name__]
    doc = from_geo(geojson, image, name=name, tilesize=tilesize, match=match,
                   default_label=default_label, label_field=label_field,
                   workers=workers, cache_type=cache_type,
                   dtype=dtype, description=description,
                   mltype=mltype, public=public, sensors=sensors,
                   partition=partition, mask=mask,
                   url=url, conn=conn, **kwargs)
    return VedaCollectionProxy.from_doc(doc)

create_from_tarball = from_tarball

def create_MLImage(cat_id, pansharpen=False, **kwargs):
    return(MLImage(cat_id,pansharpen))
