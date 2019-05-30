import logging
import os
from pyveda.auth import Auth

PROD = "https://veda-api.geobigdata.io"
DEV = "https://veda-api-development.geobigdata.io"
LOCAL = "http://host.docker.internal:3002"

DATASET_ROOT = os.environ.get(
    'PYVEDA_DATASET_ROOT',
    os.path.join(os.path.expanduser('~'), '.pyveda', 'dataset'))


def get_dataset_root():
    """Gets the path to the root directory to download and cache datasets.
    Returns:
        str: The path to the dataset root directory.
    """
    return DATASET_ROOT


def set_dataset_root(path):
    """Sets the root directory to download and cache datasets.
    There are two ways to set the dataset root directory. One is by setting the
    environment variable ``PYVEDA_DATASET_ROOT``. The other is by using this
    function. If both are specified, one specified via this function is used.
    The default dataset root is ``$HOME/.pyveda/dataset``.
    Args:
        path (str): Path to the new dataset root directory.
    """
    global DATASET_ROOT
    _dataset_root = path


def get_dataset_directory(dataset_name, create_directory=True):
    """Gets the path to the directory of given dataset.
    The generated path is just a concatenation of the global root directory
    (see :func:`set_dataset_root` for how to change it) and the dataset name.
    The dataset name can contain slashes, which are treated as path separators.
    Args:
        dataset_name (str): Name of the dataset.
        create_directory (bool): If True (default), this function also creates
            the directory at the first time. If the directory already exists,
            then this option is ignored.
    Returns:
        str: Path to the dataset directory.
    """
    path = os.path.join(DATASET_ROOT, dataset_name)
    if create_directory:
        try:
            os.makedirs(path)
        except OSError:
            if not os.path.isdir(path):
                raise
    return path


class _Config:
    def __init__(self, defhost=PROD):
        self._HOST = os.environ.get("SANDMAN_API", defhost)
        try:
            self._CONN = Auth().gbdx_connection
        except Exception as err:
            logging.error(f'Error creating connection: `{err}`')

    @property
    def HOST(self):
        return self._HOST

    @property
    def CONN(self):
        return self._CONN


config = _Config()

def set_prod():
    config._HOST = PROD

def set_dev():
    config._HOST = DEV

def set_local():
    config._CONN = Auth(oauth=False).gbdx_connection
    config._HOST = os.environ.get('SANDMAN_API', LOCAL)

def set_host(host):
    config._HOST = host

def set_conn(conn):
    config._CONN = conn




class VedaConfig:

    @property
    def host(self):
        return config.HOST

    @property
    def conn(self):
        return config.CONN

