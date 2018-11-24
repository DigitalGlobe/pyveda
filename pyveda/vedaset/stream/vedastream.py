import asyncio
import collections
import queue
import threading

from pyveda.fetch.aiohttp.client import VedaStreamFetcher
from pyveda.vedaset import BaseVedaSet, BaseVedaGroup, BaseVedaSequence

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()


class StreamingVedaSequence(BaseVedaSequence):
    def __init__(self, buf):
        self.buf = buf

    def __len__(self):
        return len(self.buf)

    def __iter__(self):
        return iter(self.buf)

    def __getitem__(self, idx):
        return self.buf[idx]

    def __next__(self):
        raise NotImplementedError("StreamingVedaSequences can only be consumed at the StreamingVedaGroup level")
#        while True:
#            try:
#                return self.buf.popleft()
#            except IndexError:
#                break
#        raise StopIteration


class StreamingVedaGroup(BaseVedaGroup):
    def __init__(self, allocated, vset):
        self.allocated = allocated
        self._num_consumed = 0
        self._vset = vset

    def __iter__(self):
        return iter(self._vset._buf)

    def __getitem__(self, idx):
        return self._vset._buf[idx]

    def __next__(self):
        while self._num_consumed < self.allocated:
            # The following get() blocks, as it should, when we're waiting for
            # the thread running the asyncio loop to fetch more data while the
            # source generator is not yet exhausted
            dp = self._vset._q.get()
            f = asyncio.run_coroutine_threadsafe(self._vset._fetcher.consume(reqs=[next(self._vset._gen)]), loop=self._vset._loop)
            self._num_consumed += 1
            return dp
        raise StopIteration

    @property
    def images(self):
        images, _ = zip(*self._vset._buf)
        return DataSequence(self, images)

    @property
    def labels(self):
        _, labels = zip(*self._vset._buf)
        return DataSequence(self, labels)


class VedaStream(BaseVedaSet):
    def __init__(self, mltype, classes, gen, partition, bufsize, cachetype=collections.deque, loop=None):
        self.partition = partition
        self._mltype = mltype
        self._classes = classes
        self._gen = gen
        self._bufsize = bufsize
        self._q = queue.Queue()
        if cachetype is collections.deque:
            self._buf = cachetype(maxlen=bufsize)
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._lock = asyncio.Lock(loop=loop)
        self._fetcher = VedaStreamFetcher(self)
        self._thread = threading.Thread(target=partial(self._producer.run_loop, loop=loop))
        self._thread.start()
        time.sleep(0.5) # Give the thread time to start, as well as the loop it's running inside

    def _initialize_buffer(self):
        reqs = []
        while len(reqs) < self._bufsize:
            reqs.append(next(self._gen))
        f = asyncio.run_coroutine_threadsafe(self._fetcher._initialize(reqs=reqs, loop=self._loop), loop=self._loop)

    @property
    def mltype(self):
        return self._mltype

    @property
    def classes(self):
        return self._classes

    @property
    def train(self):
        pass

    @property
    def test(self):
        pass

    @property
    def validate(self):
        pass


