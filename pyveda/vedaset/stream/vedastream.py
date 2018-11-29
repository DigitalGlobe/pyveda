import asyncio
import collections
import queue
import threading
import time
from functools import partial

from pyveda.fetch.aiohttp.client import VedaStreamFetcher
from pyveda.fetch.handlers import NDImageHandler, ClassificationHandler, SegmentationHandler, ObjDetectionHandler
from pyveda.vedaset.abstract import BaseVedaSet, BaseVedaGroup, BaseVedaSequence

class VSGenWrapper(object):
    def __init__(self, vs, _iter):
        self.vs = vs
        self.source = _iter
        self.stored = False
        self._n = 0

    def __iter__(self):
        return self

    def __nonzero__(self):
        if self.stored:
            return True
        try:
            self.value = next(self.source)
            self.stored = True
        except StopIteration:
            return False
        return True

    def __next__(self):
        if self.stored:
            self.stored = False
            return self.value
        val =  next(self.source)
        self._n += 1
        return val


class StreamingVedaSequence(BaseVedaSequence):
    def __init__(self, buf):
        self.buf = buf

    def __len__(self):
        return len(self.buf)

    def __iter__(self):
        return self.buf.__iter__()

    def __getitem__(self, idx):
        return self.buf[idx]


class StreamingVedaGroup(BaseVedaGroup):
    def __init__(self, allocated, vset):
        self.allocated = allocated
        self._n_consumed = 0
        self._vset = vset
        self._exhausted = False
        if allocated == 0:
            self.exhausted = True

    def __len__(self):
        return len(self._vset._buf)

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        return [self.images[idx], self.labels[idx]]

    def __next__(self):
        while self._n_consumed < self.allocated:
            try:
                nreqs = next(self._vset._gen)
            except StopIteration:
                pass
            else:
                asyncio.run_coroutine_threadsafe(self._vset._fetcher.produce_reqs(reqs=[nreqs]),
                                                 loop=self._vset._loop)

            # The following get() blocks, as it should, when we're waiting for
            # the thread running the asyncio loop to fetch more data while the
            # source generator is not yet exhausted
            dps = self._vset._q.get()
            self._n_consumed += 1
            return dps

        self.exhausted = True
        raise StopIteration

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        self._exhausted = val
        if val:
            self._vset._on_group_exhausted()

    @property
    def images(self):
        _, imgs = zip(*self._vset._buf)
        return StreamingVedaSequence(imgs)

    @property
    def labels(self):
        lbls, _ = zip(*self._vset._buf)
        return StreamingVedaSequence(lbls)


class VedaStream(BaseVedaSet):
    _lbl_handler_map = {"classification": ClassificationHandler,
                       "segmentation": SegmentationHandler,
                       "object_detection": ObjDetectionHandler}

    def __init__(self, mltype, classes, count, gen, partition, bufsize,
                 image_shape, cachetype=collections.deque, auto_startup=False,
                 auto_shutdown=False, fetcher=None, loop=None):
        self.partition = partition
        self.count = count
        self.image_shape = image_shape
        self._mltype = mltype
        self._classes = classes
        self._gen = VSGenWrapper(self, gen)
        self._bufsize = bufsize
        self._auto_startup = auto_startup
        self._auto_shutdown = auto_shutdown
        self._exhausted = False
        self._train = None
        self._test = None
        self._validate = None

        self._fetcher = fetcher
        self._loop = loop
        self._q = queue.Queue()
        self._buf = cachetype(maxlen=bufsize)
        self._thread = None

        self._img_handler_class = NDImageHandler
        self._lbl_handler_class = self._lbl_handler_map[self._mltype]

        if auto_startup:
            self._start_consumer()

    @classmethod
    def from_vc(cls, vc, count=None, bufsize=50, cachetype=collections.deque,
                auto_startup=False, auto_shutdown=False, fetcher=None, loop=None):
        if not count:
            count = vc.count
        if count > vc.count:
            raise ValueError('Input count must be less than or equal to total size of input VedaCollection instance')
        return cls(mltype=vc.mltype, classes=vc.classes, count=count,
                          gen=vc.ids(), partition=vc.partition, bufsize=bufsize,
                          image_shape=vc.imshape, cachetype=cachetype,
                          auto_startup=auto_startup, auto_shutdown=auto_shutdown,
                          fetcher=fetcher, loop=loop)


    def __len__(self):
        return self._bufsize # Whatever for now

    @property
    def mltype(self):
        return self._mltype

    @property
    def classes(self):
        return self._classes

    @property
    def train(self):
        if not self._train:
            self._train = StreamingVedaGroup(round(self.count*self.partition[0]*0.01), self)
        return self._train

    @property
    def test(self):
        if not self._test:
            self._test = StreamingVedaGroup(round(self.count*self.partition[1]*0.01), self)
        return self._test

    @property
    def validate(self):
        if not self._validate:
            self._validate = StreamingVedaGroup(round(self.count*self.partition[2]*0.01), self)
        return self._validate

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        if val:
            self._on_exhausted()
        self._exhausted = val

    def _on_group_exhausted(self):
        if all([self.train.exhausted, self.test.exhausted, self.validate.exhausted]):
            self.exhausted = True

    def _on_exhausted(self):
        if self._auto_shutdown:
            self._stop_consumer()

    def _initialize_buffer(self):
        reqs = []
        while len(reqs) < self._bufsize:
            try:
                reqs.append(next(self._gen))
            except StopIteration:
                break

        f = asyncio.run_coroutine_threadsafe(self._fetcher.produce_reqs(reqs=reqs), loop=self._loop)
        f.result()

    def _configure_fetcher(self, **kwargs):
        img_py_h = self._img_handler_class._payload_handler
        lbl_py_h = partial(self._lbl_handler_class._payload_handler,
                           klasses=self.classes, out_shape=self.image_shape)

        self._fetcher = VedaStreamFetcher(self,
                                          total_count=self.count,
                                          img_payload_handler=img_py_h,
                                          lbl_payload_handler=lbl_py_h,
                                          **kwargs)

    def _configure_worker(self, fetcher=None, loop=None):
        if not self._fetcher:
            self._fetcher = self._configure_fetcher()
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(target=partial(self._fetcher.run_loop, loop=loop))

    def _start_consumer(self):
        if not self._fetcher:
            self._configure_fetcher()
        if not self._thread:
            self._configure_worker()

        self._thread.start()
        time.sleep(0.5)
        self._consumer_fut = asyncio.run_coroutine_threadsafe(self._fetcher.start_fetch(self._loop),
                                                              loop=self._loop)
        self._initialize_buffer()

    def _stop_consumer(self):
        f = asyncio.run_coroutine_threadsafe(self._fetcher.kill_workers(), loop=self._loop)
        f.result() # Wait for workers to shutdown gracefully
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()
