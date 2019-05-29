import os
from contextlib import contextmanager

from pyveda.config import VedaConfig
from pyveda.exceptions import RemoteCollectionNotFound
from pyveda.vedaset import VedaBase, VedaStream
from pyveda.vedaset.veda.loaders import from_geo, from_tarball
from pyveda.io.io import build_vedabase
from pyveda.vedaset.veda.api import _bec, VedaCollectionProxy
from pyveda.models import Model

cfg = VedaConfig()

__all__ = ["search",
           "open",
           "store",
           "from_id",
           "from_name",
           "create_from_geojson",
           "create_from_tarball",
           "model_from_id"]

def _map_contains_submap(mmap, submap, hard_match=True):
    """ Checks if a submap is contained in a master map.
    Map values must define __eq__.

    Args:
        mmap(dict): master dictionary to check
        submap(dict): dictionary to look for
        hard_match(bool): if true, all submap keys must be present in the master map

    Returns:
        bool: True if submap keys map to values equal to master mapping
    """
    mset, sset = set(mmap.keys()), set(submap.keys())
    if hard_match and not mset.issuperset(sset):
        return False
    shared = mset.intersection(sset)
    if not shared:
        return True
    return all([mmap[key] == submap[key] for key in shared])


def search(params={}, filters={}, **kwargs):
    ''' Search Veda for collections

    Args:
        params (dict): Additional search params for Elasticsearch (optional)
        filters (dict): additional filters to apply to results (optional)

    Returns:
        list: VedaCollectionProxies for collections returned by the search
    '''
    r = cfg.conn.post('{}/{}'.format(cfg.host, "search"), json=params)
    r.raise_for_status()
    results = r.json()
    return [VedaCollectionProxy.from_doc(s) for s in results
            if _map_contains_submap(s["properties"], filters, **kwargs)]

def model_from_id(model_id):
    """
      Initialize a Model class from an id

      Args:
          model_id (str): the id of the model

      Returns:
          model (Model): the pyveda model class for the tis
    """
    return Model.from_id(model_id)

def from_id(dataset_id):
    ''' Returns the dataset as a VedaCollectionProxy from an ID if it exists.

    Args:
        dataset_id (str): ID of the dataset to check
    Returns:
        VedaCollectionProxy: the dataset
    '''

    r = cfg.conn.get(_bec._dataset_base_furl.format(host_url=cfg.host,
                                                dataset_id=dataset_id))
    r.raise_for_status()
    if r.status_code == 200:
        return VedaCollectionProxy.from_doc(r.json())
    else:
        raise Exception('Invalid dataset id, does not exist in the database.')

def from_name(dataset_name):
        ''' Returns the dataset as a VedaCollectionProxy from a name if it exists.

        Args:
            dataset_name (str): name of the dataset to check
            conn (Oauth2 connection): server connection to use

        Returns:
            VedaCollectionProxy: the dataset
        '''
        results = search(filters={"name": dataset_name})
        if results:
            return results[0]
        else:
            raise ValueError("Must provide dataset_id or dataset_name arguments")

def open(dataset_id=None, dataset_name=None, filename=None,
         partition=[70,20,10], **kwargs):
    """
    Main interface to access to remote, local and synced datasets

    Args:
      dataset_id (str): A valid dataset id for an existing collection
      dataset_name (str): A name of an existing collection
      filename (str): A local filename for a sync'd collection (created via store)
      partition (list): A list of partition percentages for train, test, validate partitions

    Returns:
      Either an intance of VedaStream (via dataset_id or dataset_name) or VedaBase (when filename is not None)
    """

    if all(v is None for v in [dataset_id, dataset_name, filename]):
        raise ValueError("When calling pyveda.open, specify one of: dataset_id, dataset_name, or filename")
    # Check for dataset on veda
    vcp = False
    if dataset_id:
        #vcp = dataset_exists(dataset_id=dataset_id)
        vcp = from_id(dataset_id)
    elif dataset_name:
        #vcp = dataset_exists(dataset_name=dataset_name)
        vcp = from_name(dataset_name)
    if vcp:
        return _load_stream(vcp, partition=partition, **kwargs)
    if filename:
        return _load_store(filename, **kwargs)
    raise RemoteCollectionNotFound("No Collection found on Veda for identifier: {}".format(dataset_id or dataset_name))


def store(filename, dataset_id=None, dataset_name=None, count=None,
          partition=[70,20,10], overwrite=False, **kwargs):
    """ Download a collection locally into a VedaBase hdf5 store

    Args:
        filename(str): Name of target hdf5 file
        dataset_id(str): ID of dataset
        dataset_name(str): Name of dataset, if ID is not used
        count(int): Number of items to store, default is None (store all)
        partition[list of int]: Percentages of datapoints to allocate to [train,test,validate] groups

    Returns:
        vedabase
    """

    if all(v is None for v in [dataset_id, dataset_name]):
        raise ValueError("When calling pyveda.store, specify one of: dataset_id, dataset_name")
    if dataset_id:
        coll = from_id(dataset_id=dataset_id)
    if dataset_name:
        coll = from_name(dataset_name = dataset_name)
    count = count or coll.count
    vb = VedaBase(filename,
                  overwrite=overwrite,
                  mltype=coll.mltype,
                  classes=coll.classes,
                  image_shape=coll.imshape,
                  image_dtype=coll.dtype,
                  partition=partition,
                  count=count,
                  dataset_id = dataset_id,
                  **kwargs)

    urlgen = coll.gen_sample_ids(count=count)
    build_vedabase(vb, urlgen, **kwargs)
    return vb


def _load_stream(vc, *args, **kwargs):
    ''' Opens a Veda collection from the server

    Args:
        vc: a VedaCollection instance

    Returns:
        VedaStream
    '''
    return VedaStream.from_vc(vc, *args, **kwargs)


def _load_store(filename, **kwargs):
    ''' Opens a Veda collection from a local hdf5 file

    Args:
        filename(str): Path to the hdf5 file

    Returns:
        VedaBase
    '''
    return VedaBase.from_path(filename, **kwargs)

# Required: tilesize, mltype,
def create_from_geojson(geojson, image, name, tilesize=[256,256], match="INTERSECT",
                              default_label=None, label_field=None,
                              workers=1, cache_type="stream",
                              dtype=None, description='',
                              mltype="classification", public=False,
                              partition=[100,0,0], mask=None,
                              background_ratio=0.0,
                              **kwargs):
    """ Loads geojson and an image into a new collection of data

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

    Returns:
        VedaCollectionProxy
    """
    assert isinstance(name, str), ValueError('Name must be defined as a string')

    sensors = [image.__class__.__name__]
    doc = from_geo(geojson, image, name=name, tilesize=tilesize, match=match,
                   default_label=default_label, label_field=label_field,
                   workers=workers, cache_type=cache_type,
                   dtype=dtype, description=description,
                   mltype=mltype, public=public, sensors=sensors,
                   partition=partition, mask=mask,
                   background_ratio=background_ratio,
                   **kwargs)
    return VedaCollectionProxy.from_doc(doc)

def create_from_tarball(s3path, name, mltype="classification", imshape=[3,256,256]):
    ''' Creates a new collection from tarball

    Args:
        s3path (str): The url to the tarball.
        name (str): A name for the collection.
        mltype (str): The type model this data may be used for training. One of 'classification', 'object detection', 'segmentation'.
        imshape (list): Shape of image data. Multiband should be X,N,M. Single band should be 1,N,M.

    Returns:
        VedaCollectionProxy
    '''
    assert isinstance(name, str), ValueError('Name must be defined as a string')
    doc = from_tarball(s3path, name=name, mltype=mltype, imshape=imshape)
    return VedaCollectionProxy.from_doc(doc)
