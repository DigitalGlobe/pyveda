import asyncio
from collections import deque
from concurrent.futures import CancelledError
from queue import Empty, Full, Queue
import threading

from pyveda.vedaset import BaseVedaSet, BaseVedaGroup, BaseVedaSequence

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()


class VedaStreamFetcher(BatchFetchTracer):
    def __init__(self, total_count=0, session=None, token=None, max_retries=5, timeout=20,
                 session_limit=30, max_concurrent_requests=200, connector=aiohttp.TCPConnector,
                 lbl_payload_handler=lambda x: x, img_payload_handler=lambda x: x, write_fn=lambda x: x,
                 img_batch_transform=lambda x: x, lbl_batch_transform=lambda x: x,
                 num_lbl_payload_threads=1, num_img_payload_threads=10, num_write_workers=1, num_write_threads=1,
                 lbl_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 img_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 write_executor=concurrent.futures.ThreadPoolExecutor,
                 run_tracer=False, *args, **kwargs):

        self.max_concurrent_reqs = min(total_count, max_concurrent_requests)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session
        self._token = token
        self._session_limit = session_limit
        self._connector = connector
        self._trace_configs = []
        self._run_tracer = run_tracer
        if run_tracer:
            trace_config = self._configure_tracer()
            self._trace_configs.append(trace_config)

        self.write_fn = write_fn
        self._lbl_batch_transform = lbl_batch_transform
        self._img_batch_transform = img_batch_transform
        self._n_lbl_payload_threads = num_lbl_payload_threads
        self._n_img_payload_threads = num_img_payload_threads
        self._lbl_payload_executor = lbl_payload_executor(max_workers=num_lbl_payload_threads)
        self._img_payload_executor = img_payload_executor(max_workers=num_img_payload_threads)
        self._n_write_workers = num_write_workers
        self._n_write_threads = num_write_threads
        self._write_executor = write_executor(max_workers=num_write_threads)

        self.lbl_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._lbl_payload_executor,
                                                     fn=lbl_payload_handler)
        self.img_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._img_payload_executor,
                                                     fn=img_payload_handler)
    @property
    def headers(self):
        if not self._token:
            raise AttributeError("Provide an auth token on init or as an argument to run")
        return {"Authorization": "Bearer {}".format(self._token)}

    async def _payload_handler(self, payload, executor=None, fn=lambda x: x):
        processed_data = await self.loop.run_in_executor(executor, fn, payload)
        return processed_data

    async def fetch_with_retries(self, url, json=True, callback=None, **kwargs):
        await asyncio.sleep(0.0)
        retries = self.max_retries
        while retries:
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    #logger.info("  URL READ SUCCESS: {}".format(url))
                    if json:
                        data = await response.json()
                    else:
                        data = await response.read()
                    await response.release()
                if callback:
                    try:
                        data = await callback(data, **kwargs)
                    except Exception as e:
                        #logger.info(e)
                return data
            except CancelledError:
                break
            except Exception as e:
                #logger.info(e)
                #logger.info("    URL READ ERROR: {}".format(url))
                retries -= 1
        return None

    async def write_stack(self):
        await asyncio.sleep(0.0)
        while True:
            try:
                item = await self._qwrite.get()
                async with self._write_lock:
                    self.runner.q.put_nowait(item)
                self._qreq.task_done()
                self._qwrite.task_done()
            except CancelledError: # write out anything remaining
                break
        return True

    async def consume_reqs(self):
        while True:
            try:
                label_url, image_url = await self._qreq.get()
                flbl = asyncio.ensure_future(self.fetch_with_retries(label_url, callback=self.lbl_payload_handler))
                fimg = asyncio.ensure_future(self.fetch_with_retries(image_url, json=False, callback=self.img_payload_handler))
                label, image = await asyncio.gather(flbl, fimg)
                await self._qwrite.put([image, label])
            except CancelledError:
                break

    async def produce_reqs(self, reqs=None):
        for req in reqs:
            if req is None:
                await self._worker.cancel()
                return True
            await self._qreq.put(req)
        self._source_exhausted.set()
        return True

    def _configure(self, session, loop):
        self.session = session
        self.loop = loop
        self._source_exhausted = asyncio.Event()
        self._qreq = asyncio.Queue(maxsize=self.max_concurrent_reqs)
        self._qwrite = asyncio.Queue()
        self._write_lock = asyncio.Lock()
        self._consumers = [asyncio.ensure_future(self.consume_reqs()) for _ in range(self.max_concurrent_reqs)]
        self._writers = [asyncio.ensure_future(self.write_stack()) for _ in range(self._n_write_workers)]

    async def kill_workers(self):
        await self._source_exhausted.wait()
        await self._qwrite.join()
        for fut in self._consumers:
            fut.cancel()
        for fut in self._writers:
            fut.cancel()
        done, pending = await asyncio.wait(self._writers)
        return True

    async def drive_fetch(self, session, loop):
        self._configure(session, loop)
        producer = await self.produce_reqs()
        res = await self.kill_workers()

    async def start_fetch(self, loop):
        async with aiohttp.ClientSession(loop=loop,
                                         connector=self._connector(limit=self._session_limit),
                                         headers=self.headers,
                                         trace_configs=self._trace_configs) as session:
            logger.info("BATCH FETCH START")
            results = await self.drive_fetch(session, loop)
            logger.info("BATCH FETCH COMPLETE")
            return results

    def run_loop(self, reqs=None, token=None, loop=None):
        if not loop:
            loop = asyncio.get_event_loop()
        self.loop = loop
        if reqs:
            self.reqs = reqs
        if token:
            self._token = token
        asyncio.set_event_loop(loop)
#        fut = await self.start_fetch(loop)
        loop.run_forever()


class StreamingVedaSequence(BaseVedaSequence):
    def __init__(self, buf):
        assert isinstance(buf, collections.deque)
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
        self._producer = AsyncProducer(self)

    def __iter__(self):
        while True:
            while self._num_consumed < self.allocated:
                dps = self._vset.get()
                self.buf.append(dps)
                f = asyncio.run_coroutine_threadsafe(self._producer.consume(), loop=self._vset._loop)
                return dp

    def __getitem__(self, idx):
        return self.buf[idx]

    def __next__(self):
        while self._num_consumed < self.allocated:
            # The following get() blocks, as it should, when we're waiting for
            # the thread running the asyncio loop to fetch more data while the
            # source generator is not yet exhausted
            dp = self._vset._q.get()
            f = asyncio.run_coroutine_threadsafe(self._vset._fetcher.consume(reqs=[next(self._vset._gen)]), loop=self._vset._loop)
            self._num_consumed += 1
            return dp

    @property
    def images(self):
        images, _ = zip(*self._vset._buf)
        return DataSequence(self, images)

    @property
    def labels(self):
        _, labels = zip(*self._vset._buf)
        return DataSequence(self, labels)


class VedaStream(BaseVedaSet):
    def __init__(self, mltype, gen, partition, bufsize, cachetype=collections.deque, loop=None):
        self.partition = partition
        self._mltype = mltype
        self._gen = gen
        self._bufsize = bufsize
        self._q = queue.Queue()
        if cachetype is collections.deque:
            self._buf = cachetype(maxlen=bufsize)
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._lock = asyncio.Lock(loop=loop)
        self._producer = AsyncStreamer(self)
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
    def train(self):
        pass

    @property
    def test(self):
        pass

    @property
    def validate(self):
        pass















