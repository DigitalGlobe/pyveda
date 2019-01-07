import asyncio
import collections
import queue
import threading
import time
from functools import partial

from pyveda.fetch.aiohttp.client import VedaStreamFetcher
from pyveda.fetch.handlers import NDImageHandler, ClassificationHandler, SegmentationHandler, ObjDetectionHandler
from pyveda.vedaset.abstract import BaseVariableArray, BaseSampleArray, BaseDataSet

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


class BufferedVariableArray(BaseVariableArray):
    def __init__(self, buf):
        self.buf = buf

    def __len__(self):
        return len(self.buf)

    def __iter__(self):
        return self.buf.__iter__()

    def __getitem__(self, idx):
        return self.buf[idx]


class BufferedSampleArray(BaseSampleArray):
    def __init__(self, allocated, vset):
        self.allocated = allocated
        self._n_consumed = 0
        self._vset = vset
        self._exhausted = False
        if allocated == 0:
            self.exhausted = True

    def __len__(self):
        return self.allocated

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
        try:
            _, imgs = zip(*self._vset._buf)
        except ValueError:
            imgs = []
        return BufferedVariableArray(imgs)

    @property
    def labels(self):
        try:
            lbls, _ = zip(*self._vset._buf)
        except ValueError:
            lbls = []
        return BufferedVariableArray(lbls)


class BufferedDataStream(BaseDataSet):
    _lbl_handler_map = {"classification": ClassificationHandler,
                       "segmentation": SegmentationHandler,
                       "object_detection": ObjDetectionHandler}

    def __init__(self, mltype, classes, _count, gen, image_shape,
                 partition=[70, 20, 10], bufsize=100, cachetype=collections.deque,
                 auto_startup=False, auto_shutdown=False, fetcher=None, loop=None, **kwargs):
        self.partition = partition
        self.count = _count
        self.image_shape = image_shape
        self._mltype = mltype
        self._classes = classes
        self._gen = VSGenWrapper(self, gen)
        self._auto_startup = auto_startup
        self._auto_shutdown = auto_shutdown
        self._exhausted = False
        self._train = None
        self._test = None
        self._validate = None
        self._bufsize = bufsize if bufsize < self.count else self.count

        self._fetcher = fetcher
        self._loop = loop
        self._q = queue.Queue()
        self._buf = cachetype(maxlen=bufsize)
        self._thread = None

        self._img_handler_class = NDImageHandler
        self._lbl_handler_class = self._lbl_handler_map[self._mltype]

        if auto_startup:
            self._start_consumer()


    def __len__(self):
        return self.count

    @property
    def mltype(self):
        return self._mltype

    @property
    def classes(self):
        return self._classes

    @property
    def train(self):
        if not self._train:
            self._train = BufferedSampleArray(round(self.count*self.partition[0]*0.01), self)
        return self._train

    @property
    def test(self):
        if not self._test:
            self._test = BufferedSampleArray(round(self.count*self.partition[1]*0.01), self)
        return self._test

    @property
    def validate(self):
        if not self._validate:
            self._validate = BufferedSampleArray(round(self.count*self.partition[2]*0.01), self)
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
            self._configure_fetcher()
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(target=partial(self._fetcher.run_loop, loop=loop))

    def _start_consumer(self, init_buff=True):
        if not self._fetcher:
            self._configure_fetcher()
        if not self._thread:
            self._configure_worker()

        self._thread.start()
        time.sleep(0.5)
        self._consumer_fut = asyncio.run_coroutine_threadsafe(self._fetcher.start_fetch(self._loop),
                                                              loop=self._loop)
        if init_buff:
            self._initialize_buffer() # Fill the buffer and block until full

    def _stop_consumer(self):
        self._consumer_fut.cancel()
        f = asyncio.run_coroutine_threadsafe(self._fetcher.kill_workers(), loop=self._loop)
        f.result() # Wait for workers to shutdown gracefully
        for task in asyncio.Task.all_tasks():
            task.cancel()
        self._loop.create_task(self._fetcher.session.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    @classmethod
    def from_vc(cls, vc, **kwargs):
        return cls(vc.mltype, vc.classes, vc.count, vc.gen_sample_ids(**kwargs),
                   vc.imshape, **kwargs)

    def __enter__(self):
        self._start_consumer()
        return self

    def __exit__(self, *args):
        self._stop_consumer()

    def __getitem__(self, slc):
        raise NotImplementedError



