#from pyveda.vedaset.util import _dataset_exists_in_veda
import os
from pyveda.exceptions import RemoteCollectionNotFound
from pyveda.auth import Auth
from pyveda.vedaset import VedaBase
from pyveda.datapoint import DataPoint
from pyveda.veda.api import search
from pyveda.veda.api import VedaCollectionProxy

gbdx = Auth()
HOST = os.environ.get('SANDMAN_API', "https://veda-api.geobigdata.io")
conn = gbdx.gbdx_connection

def load(dataset_id=None, dataset_name=None, filename=None):
    if not dataset_id or dataset_name or filename:
        raise ValueError("When calling pyveda.load, specify one of: dataset_id, dataset_name, or filename")
    # Check for dataset on veda
    identifier = dataset_id or dataset_name
    if identifier:
        dataset_id = _dataset_exists_in_veda(identifier)
        if dataset_id:
            return load_existing(dataset_id)
    if filename:
        return load_store(filename)
    raise RemoteCollectionNotFound("No Collection found on Veda for identifier: {}".format(identifier))


def load_existing(*args, **kwargs):
    pass

def load_streamer(*args, **kwargs):
    pass

def load_store(*args, **kwargs):
    pass

def _dataset_exists_in_veda(*args, **kwargs):
    pass


