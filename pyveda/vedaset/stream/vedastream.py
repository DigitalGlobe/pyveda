import asyncio
import collections
import queue
import threading

from pyveda.fetch.aiohttp.client import VedaStreamFetcher
from pyveda.vedaset.abstract import BaseVedaSet, BaseVedaGroup, BaseVedaSequence

class VSGenWrapper(object):
    def __init__(self, vs, _iter):
        self.vs = vs
        self.source = _iter
        self.stored = False

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
        return next(self.source)


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

    def __iter__(self):
        return self

    def __getitem__(self, idx):
        pass
#        return self._vset._buf[idx]

    def __next__(self):
        while self._n_consumed < self.allocated:
            # The following get() blocks, as it should, when we're waiting for
            # the thread running the asyncio loop to fetch more data while the
            # source generator is not yet exhausted
            dp = self._vset._q.get()
            try:
                nreq = next(self._vset._gen) # This ought to throw after the last get()
            except StopIteration:
                self._vset._on_exhausted()
            else:
                f = asyncio.run_coroutine_threadsafe(self._vset._fetcher.consume(reqs=[nreq]), loop=self._vset._loop)
            finally:
                self._n_consumed += 1
                return dp

        self._exhausted = True
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
        images, _ = zip(*self._vset._buf)
        return DataSequence(self, list(images))

    @property
    def labels(self):
        _, labels = zip(*self._vset._buf)
        return DataSequence(self, list(labels))


class VedaStream(BaseVedaSet):
    def __init__(self, mltype, classes, count, gen, partition, bufsize, cachetype=collections.deque, loop=None):
        self.partition = partition
        self.count = count
        self._mltype = mltype
        self._classes = classes
        self._gen = VSGenWrapped(self, gen)
        self._bufsize = bufsize
        self._exhausted = False
        self._train = None
        self._test = None
        self._validate = None

        self._q = queue.Queue()
        if cachetype is collections.deque:
            self._buf = cachetype(maxlen=bufsize)
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._lock = asyncio.Lock(loop=loop)
        self._fetcher = VedaStreamFetcher(self)
        self._thread = threading.Thread(target=partial(self._fetcher.run_loop, loop=loop))
        self._thread.start()
        time.sleep(0.5) # Give the thread time to start, as well as the loop it's running inside

        self._initialize_buffer()

    def _initialize_buffer(self):
        reqs = []
        while len(reqs) < self._bufsize:
            reqs.append(next(self._gen))
        f = asyncio.run_coroutine_threadsafe(self._fetcher._initialize(reqs=reqs, loop=self._loop), loop=self._loop)

    def _on_group_exhausted(self):
        if not any(self.train.exhausted, self.test.exhausted, self.validate.exhausted):
            self._exhausted = True

    def _on_exhausted(self):
        f = asyncio.run_coroutine_threadsafe(self._fetcher.kill_workers(), loop=self._loop)
        f.result() # Wait for workers to shutdown gracefully
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        if val:
            self._on_exhausted()
        self._exhausted = val

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


