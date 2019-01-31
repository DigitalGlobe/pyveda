import asyncio
import collections
import queue
import threading
import time
from functools import partial
import numpy as np

from pyveda.fetch.aiohttp.client import VedaStreamFetcher
from pyveda.vedaset.abstract import BaseVariableArray, BaseSampleArray, BaseDataSet


class BufferedVariableArray(BaseVariableArray):
    pass


class BufferedSampleArray(BaseSampleArray):
    def __init__(self, vset):
        super(BufferedSampleArray, self).__init__(vset)
        self._n_consumed = 0
        self._exhausted = False
        if self.allocated == 0:
            self.exhausted = True

    def __len__(self):
        return self.allocated

    def __iter__(self):
        return self

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

    def batch_iter(self, batch_size):
        while True:
            batch = []
            while len(batch) < batch_size:
                batch.append(self.__next__())
            yield batch

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        self._exhausted = val
        if val:
            self._vset._on_group_exhausted()


class BufferedDataStream(BaseDataSet):
    _fetch_class = VedaStreamFetcher
    _sample_class = BufferedSampleArray
    _variable_class = BufferedVariableArray

    def __init__(self, source, bufsize=100, auto_startup=False,
                 auto_shutdown=False, write_index=True, write_h5=False,
                 *args, **kwargs):
        super(BufferedDataStream, self).__init__(*args, **kwargs)
        self._gen = source
        self._auto_startup = auto_startup
        self._auto_shutdown = auto_shutdown
        self._exhausted = False
        self._bufsize = bufsize if bufsize < self.count else self.count

        self._fetcher = None
        self._loop = None
        self._q = queue.Queue()
        self._buf = collections.deque(maxlen=bufsize)
        self._thread = None

        self._write_index = write_index
        self._write_h5 = write_h5

        if auto_startup:
            self._start_consumer()

    @property
    def _img_arr(self):
        _, imgs, _ = zip(*self._buf)
        return self._variable_class(self, imgs)

    @property
    def _lbl_arr(self):
        lbls, _ = zip(*self._buf)
        return self._variable_class(self, lbls)

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
        time.sleep(1.0)
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

    def __len__(self):
        return self.count


