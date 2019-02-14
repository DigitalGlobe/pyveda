from pyveda.vedaset.store.vedabase import H5DataBase
from pyveda.vedaset.stream.vedastream import BufferedDataStream
from pyveda.veda.api import VedaCollectionProxy
from contextlib import ContextDecorator


class VedaBase(H5DataBase):
    pass


class VedaStream(BufferedDataStream):
    pass
