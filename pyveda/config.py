import os
from pyveda.auth import Auth

PROD = "https://veda-api.geobigdata.io"
DEV = "https://veda-api-development.geobigdata.io"
LOCAL = "http://host.docker.internal:3002"

class _Config:
    def __init__(self, defhost=PROD):
        self._HOST = os.environ.get("SANDMAN_API", defhost)
        self._CONN = Auth().gbdx_connection

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
    config._HOST = LOCAL

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
        
