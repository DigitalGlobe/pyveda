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
            return True if not return_coll else VedaCollectionProxy.from_docs(r.json())
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
        return load_existing(obj=vcp, count=count, partition=partition, **kwargs)
    if filename:
        return load_store(filename)
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

def load_existing(vid, *args, **kwargs):
    return VedaStream.from_id(vid, *args, **kwargs)

def load_store(filename):
    return VedaBase.from_path(filename)

def load_new(*args, **kwargs):
    # bulk load new datasets here
    pass


