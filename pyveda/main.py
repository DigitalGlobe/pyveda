import os
from pyveda.exceptions import RemoteCollectionNotFound
from pyveda.auth import Auth
from pyveda.vedaset import VedaBase
from pyveda.datapoint import DataPoint
from pyveda.veda.api import search, dataset_exists, VedaCollectionProxy

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection

def load(dataset_id=None, dataset_name=None, filename=None, count=None, partition=[70,20,10], **kwargs):
    """
    Main interface to access to remote, local and synced datasets
    """

    if not dataset_id or dataset_name or filename:
        raise ValueError("When calling pyveda.load, specify one of: dataset_id, dataset_name, or filename")
    # Check for dataset on veda
    identifier = dataset_id or dataset_name
    if identifier:
        dataset_id = dataset_exists(identifier)
        if dataset_id:
            return load_existing(dataset_id)
    if filename:
        return load_store(filename)
    raise RemoteCollectionNotFound("No Collection found on Veda for identifier: {}".format(identifier))


def store(dataset_id=None, dataset_name=None, count=None, partition=[70,20,10], **kwargs):
    if not dataset_id or dataset_name:
        raise ValueError("When calling pyveda.store, specify one of: dataset_id, dataset_name")
    coll = load(dataset_id=dataset_id, dataset_name=dataset_name)
    vb = VedaBase.from_coll(coll, count=count, partition=partition)
    pgen = vb.ids()

    build_vedabase(vb, gbdx.gbdx_connection.access_token, **kwargs)
    vb.flush()
    return vb

def load_existing(vid, *args, **kwargs):
    return VedaStream.from_id(vid, *args, **kwargs)

def load_streamer(*args, **kwargs):
    pass

def load_store(filename):
    return VedaBase.from_path(filename)


