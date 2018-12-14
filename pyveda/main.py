import os
from pyveda.exceptions import RemoteCollectionNotFound
from pyveda.auth import Auth
from pyveda.vedaset import VedaBase, VedaStream
from pyveda.veda.api import VedaCollectionProxy

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection

__all__ = ["search",
           "load",
           "store",
           "load_new",
           "load_store",
           "load_existing",
           "dataset_exists"]


def search(params={}, host=HOST):
    r = conn.post('{}/{}'.format(host, "search"), json=params)
    r.raise_for_status()
    try:
        results = r.json()
        return [VedaCollectionProxy.from_doc(s) for s in results]
    except Exception as err:
        print(err)
        return []

def dataset_exists(dataset_id=None, name=None, conn=conn, host=HOST):
    if dataset_id:
        r = conn.get(_bec._dataset_base_furl.format(host_url=host,
                                                    dataset_id=dataset_id))
        return True if r.status_code == 200 else False
    if name:
        r = conn.post("{}/search".format(host), json={})
        r.raise_for_status()
        return any([True for p in r.json() if p['properties']['name'] == name])
    raise ValueError("Must provide dataset_id or name arguments")


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

def load_store(filename):
    return VedaBase.from_path(filename)

def load_new(*args, **kwargs):
    # bulk load new datasets here
    pass


