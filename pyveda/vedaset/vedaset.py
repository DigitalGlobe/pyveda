from pyveda.vedaset.store.vedabase import H5DataBase
from pyveda.vedaset.stream.vedastream import BufferedDataStream
from pyveda.vedaset.veda.api import VedaCollectionProxy
from contextlib import ContextDecorator
from collections import ChainMap


class VedaSet(object):
    def __init__(self, *args, **kwargs):
        self._vs_ = None

    def __getattr__(self, attr):
        if attr not in self.__dir__():
            raise AttributeError
        pass

    @property
    def _vs(self):
        return self._vs_

    @_vs.setter
    def _vs(self, vobj):
        if not self._vs_:
            self._vs_ = vobj

    def as_stream(self, *args, **kwargs):
        pass

    def buid_store(self, background=True):
        pass

    @classmethod
    def from_proxy(cls, proxy):
        pass

    @classmethod
    def from_store(cls, store_ref):
        pass

    @classmethod
    def from_id(cls, id):
        pass

    @classmethod
    def from_name(cls, name):
        pass


