import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass
import aiohttp
import concurrent.futures
from concurrent.futures import CancelledError, TimeoutError
import threading

import numpy as np

from tempfile import NamedTemporaryFile
import os
import sys
import functools
import json
import collections
import logging
import logging.handlers

from pyveda.io.utils import write_trace_profile
from pyveda.config import VedaConfig

has_tqdm = False
try:
    from tqdm import trange, tqdm, tqdm_notebook, tnrange
    has_tqdm = True
except ImportError:
    pass

log_fh = ".{}.log".format(__name__)
logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.DEBUG)
handler = logging.handlers.RotatingFileHandler(
            log_fh, mode="w", maxBytes=10485760, backupCount=1)
handler.setFormatter(formatter)
logger.addHandler(handler)

cfg = VedaConfig()


class ThreadedAsyncioRunner(object):
    def __init__(self, run_method, call_method, loop=None):
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._method = call_method
        self._thread = threading.Thread(target=functools.partial(run_method, loop=loop))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def __call__(self, *args, **kwargs):
        f = asyncio.run_coroutine_threadsafe(self._method(*args, **kwargs), self._loop)
        return f.result()


class PhaseBuffer(object):
    def __init__(self, period=1, maxlen=None):
        self.period = period
        self._count = 0
        self._cycle_num = 1
        self._full_cycles = 0
        self._phased = False
        self._ibuf = deque(maxlen=maxlen)

    @property
    def count(self):
        return self._count

    @property
    def phased(self):
        return self._phased

    @phased.setter
    def phased(self, v):
        self._phased = v
        if v:
            self._full_cycles = self._cycle_num
            self._cycle_num += 1

    @property
    def k(self):
        return self.count % self.period

    @property
    def full_cycles(self):
        return self._full_cycles

    @property
    def cycle_number(self):
        return self.full_cycles + 1

    @property
    def current_sample(self):
        n = self.k
        if self.phased:
            n = self.period
        return list(tail(n, self.view))

    def append(self, obj):
        self._phased = False
        self._ibuf.append(obj)
        self._count += 1
        if self.k == 0:
            self.phased = True

    @property
    def view(self):
        return self._ibuf.copy()


class HTTPClientTracer(object):
    """
    A mixin class that can be used to record infomation about
    aiohttp request lifecycles.
    """

    def __init__(self, cache=None):
        if not cache:
            cache = collections.defaultdict(list)
        self._trace_cache = cache

    async def on_request_exception(self, session, context, params):
        pass

    async def on_connection_queued_start(self, session, context, params):
        pass

    async def on_connection_queued_end(self, session, context, params):
        pass

    async def on_dns_resolvehost_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_dns_resolvehost_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["dns_resolution_time"].append(elapsed)

    async def on_request_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_request_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["request_times"].append(elapsed)

    async def on_connection_create_start(self, session, context, params):
        context.start = session.loop.time()

    async def on_connection_create_end(self, session, context, params):
        elapsed = session.loop.time() - context.start
        self._trace_cache["connection_lifetimes"].append(elapsed)

    def _configure_tracer(self):
        trace_config = aiohttp.TraceConfig()
        trace_config.on_dns_resolvehost_start.append(self.on_dns_resolvehost_start)
        trace_config.on_dns_resolvehost_end.append(self.on_dns_resolvehost_end)
        trace_config.on_request_start.append(self.on_request_start)
        trace_config.on_request_end.append(self.on_request_end)
        trace_config.on_request_exception.append(self.on_request_exception)
        trace_config.on_connection_queued_start.append(self.on_connection_queued_start)
        trace_config.on_connection_queued_end.append(self.on_connection_queued_end)
        trace_config.on_connection_create_start.append(self.on_connection_create_start)
        trace_config.on_connection_create_end.append(self.on_connection_create_end)
        return trace_config


idfn = lambda x: x

def run_in_executor(fn, loop=None, executor=None):
    @wraps(fn)
    async def wrapper(*args):
        res = await loop.run_in_executor(executor, fn, *args)
        return res
    return wrapper

def run_with_lock(fn, loop=None, lock=None):
    if not lock:
        lock = asyncio.Lock(loop=loop)
    @wraps(fn)
    async def wrapper(*args):
        async with lock:
            if inspect.iscoroutinefunction(fn):
                res = await fn(*args)
            else:
                res = fn(*args)
        return res
    return wrapper


class HTTPDataClient(HTTPClientTracer):
    def __init__(self,
                 total_count=0,
                 session=None,
                 token=None,
                 max_retries=5,
                 timeout=20,
                 session_limit=30,
                 max_concurrent_requests=200,
                 max_memarrays=100,
                 suppress_callback_errors = True
                 suppress_http_errors = True
                 on_data=idfn,
                 num_write_workers=1,
                 num_write_threads=1,
                 lbl_payload_handler=idfn,
                 img_payload_handler=idfn,
                 img_batch_transform=idfn,
                 lbl_batch_transform=idfn,
                 num_lbl_payload_threads=10,
                 num_img_payload_threads=10,
                 lbl_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 img_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 write_executor=concurrent.futures.ThreadPoolExecutor,
                 connector=aiohttp.TCPConnector,
                 suppress_callback_errors = True
                 suppress_http_errors = True
                 run_tracer=False,
                 buf=None,
                 *args, **kwargs):

        self.max_concurrent_reqs = min(total_count, max_concurrent_requests)
        self.suppress_cb_errs = suppress_callback_errors
        self.suppress_http_errs = suppress_http_errors
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session
        self._total_count = total_count
        self._token = token
        self._session_limit = session_limit
        self._connector = connector
        self._trace_configs = []
        self._run_tracer = run_tracer
        if run_tracer:
            trace_config = self._configure_tracer()
            self._trace_configs.append(trace_config)

        self.on_data = on_data
        self.max_memarrs = max_memarrays
        self._lbl_batch_transform = lbl_batch_transform
        self._img_batch_transform = img_batch_transform
        self._n_lbl_payload_threads = num_lbl_payload_threads
        self._n_img_payload_threads = num_img_payload_threads
        self._n_write_workers = num_write_workers
        self._n_write_threads = num_write_threads
        self._write_executor = write_executor(max_workers=num_write_threads)
        self._lbl_payload_executor = lbl_payload_executor(max_workers=num_lbl_payload_threads)
        self._img_payload_executor = img_payload_executor(max_workers=num_img_payload_threads)


        self.lbl_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._lbl_payload_executor,
                                                     fn=lbl_payload_handler)
        self.img_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._img_payload_executor,
                                                     fn=img_payload_handler)
        if not buf:
            buf = PhaseBuffer(period=max_memarrays, maxlen=max_memarrays)
        self.data_buf = buf

        self.io_callbacks = []
        self.cpu_callbacks = []

    @property
    def headers(self):
        return {"Authorization": "Bearer {}".format(self._token)}

    async def _payload_handler(self, payload, executor=None, fn=lambda x: x):
        processed_data = await self.loop.run_in_executor(executor, fn, payload)
        return processed_data

    async def fetch_with_retries(self, url, json=True, callback=None, **kwargs):
        await asyncio.sleep(0.0)
        data = None
        retries = self.max_retries
        while retries:
            try:
                async with self.session.get(url) as response:
                    response.raise_for_status()
                    logger.info("  URL READ SUCCESS: {}".format(url))
                    if json:
                        data = await response.json()
                    else:
                        data = await response.read()
                    await response.release()
                if callback:
                    try:
                        data = await callback(data, **kwargs)
                    except Exception as e:
                        logger.info(e)
                        if not self.suppress_cb_errs:
                            raise
                return data
            except CancelledError:
                break
            except Exception as e: # TODO: catch appr aiohttp exceptions here
                logger.info(e)
                logger.info("    URL READ ERROR: {}".format(url))
                retries -= 1
                if not retries and not self.suppress_http_errs:
                    raise
        return data

    async def start_fetch(self, loop):
        async with aiohttp.ClientSession(loop=loop,
                                         connector=self._connector(limit=self._session_limit),
                                         headers=self.headers,
                                         trace_configs=self._trace_configs) as session:
            logger.info("BATCH FETCH START")
            results = await self.drive_fetch(session, loop)
            logger.info("BATCH FETCH COMPLETE")
            return results


    async def consume_reqs(self):
        while True:
            try:
                label_url, image_url, _id = await self._qreq.get()
                flbl = asyncio.ensure_future(self.fetch_with_retries(
                    label_url, callback=self.lbl_payload_handler))
                fimg = asyncio.ensure_future(self.fetch_with_retries(
                    image_url, json=False, callback=self.img_payload_handler))
                label, image = await asyncio.gather(flbl, fimg)
                await self._qwrite.put([label, image, _id])
            except CancelledError:
                break

    def _configure(self, session, loop):
        self.session = session
        self.loop = loop
        self._source_exhausted = asyncio.Event(loop=loop)
        self._qreq = asyncio.Queue(maxsize=self.max_concurrent_reqs, loop=loop)
        self._qwrite = asyncio.Queue(loop=loop)
        self._buf_lock = asyncio.Lock(loop=loop)
        self._consumers = [asyncio.ensure_future(self.consume_reqs(), loop=loop)
                           for _ in range(self.max_concurrent_reqs)]
        self._writers = [asyncio.ensure_future(self.process_data(), loop=loop)
                         for _ in range(self._n_write_workers)]

        self.run_in_exec = functools.partial(run_in_exec, loop=loop)
        self.run_with_lock = functools.partial(run_with_lock, loop=loop)

    def register_io_callback(self, fn, executor=None, lock=None):
        if not executor:
            executor = self._io_executor
        cb = functools.partial(self.run_in_exec, fn=fn, executor=executor)
        if lock:
            cb = functools.partial(self.run_with_lock, fn=fn, lock=lock)
        #self.io_callbacks.register(cb)
        self.io_callbacks.append(cb)

    def register_cpu_callback(self, fn, executor=None, lock=None):
        if not executor:
            executor = self._cpu_executor
        cb = functools.partial(self.run_in_exec, fn=fn, executor=executor)
        if lock:
            cb = functools.partial(self.run_with_lock, fn=fn, lock=lock)
        #self.cpu_callbacks.register(cb)
        self.cpu_callbacks.append(cb)

    async def kill_workers(self):
        await self._qwrite.join()
        # Next line runs io on any remaining data in the buffer
        if not self.data_buf.phased:
            await self.run_io_callbacks(self.data_buf.current_sample)
        for fut in self._consumers:
            fut.cancel()
        for fut in self._writers:
            fut.cancel()
        done, pending = await asyncio.wait(self._writers)
        return True

    def run_loop(self, loop=None):
        if not loop:
            loop = asyncio.get_event_loop()
        self.loop = loop
        asyncio.set_event_loop(loop)
        loop.run_forever()

    async def drive_fetch(self, session, loop):
        self._configure(session, loop)
        await self._source_exhausted.wait()
        res = await self.kill_workers()

    async def filter_data(self, data):
        await asyncio.sleep(0)
        return data

    async def process_data(self):
        while True:
            try:
                data = await self._qwrite.get()
                self._qreq.task_done()

                data = await self.filter_data(data)
                if not data:
                    self._qwrite.task_done()
                    continue

                dstack = None
                async with self._data_lock:
                    self.data_buf.append(data)
                    self.on_data(data)
                    if self.data_buf.phased: # Get the view, release the lock
                        dstack = self.data_buf.current_sample
                if dstack:
                    await self.run_io_callbacks(dstack)
                self._qwrite.task_done()

            except CancelledError as ce:
                break
        return True

    async def run_iocallbacks(self, dstack):
        for cb in self.io_callbacks:
            try:
                await cb(data)
            except Exception as e:
                logger.info(
                    "User exception in io callback: {}".format(e))
                if not self.suppress_cb_errs:
                    raise

    async def produce_reqs(self, reqs=[]):
        for req in reqs:
            if req is None:
                self._source_exhausted.set()
                return True
            await self._qreq.put(req)
        return True


class VedaBaseFetcher(BaseVedaSetFetcher):
    def __init__(self, reqs, **kwargs):
        self.reqs = reqs
        super(VedaBaseFetcher, self).__init__(**kwargs)
        self._pbar = None
        if has_tqdm and self._total_count:
            self._pbar = tqdm(total=self._total_count)

    async def write_stack(self):
        await asyncio.sleep(0.0)
        labels, images, _ids = [], [], []
        while True:
            try:
                label, image, _id = await self._qwrite.get()
                labels.append(label)
                images.append(image)
                _ids.append(_id)
                if len(images) == self.max_memarrs:
                    data = [self._img_batch_transform(images), self._lbl_batch_transform(labels), _ids]
                    async with self._write_lock:
                        try:
                            await self.loop.run_in_executor(self._write_executor, self.write_fn, data)
                            logger.info("SUCCESS WRITE {} DATAPOINTS".format(len(images)))
                        except Exception as e:
                    labels, images, _ids = [], [], []
                self._qreq.task_done()
                self._qwrite.task_done()
                if self._pbar:
                    self._pbar.update(1)
            except CancelledError: # write out anything remaining
                if images:
                    data = [self._img_batch_transform(images), self._lbl_batch_transform(labels), _ids]
                    async with self._write_lock:
                        await self.loop.run_in_executor(self._write_executor, self.write_fn, data)
                break
        return True

    async def drive_fetch(self, session, loop):
        self._configure(session, loop)
        producer = await self.produce_reqs()
        res = await self.kill_workers()

    async def produce_reqs(self):
        for req in self.reqs:
            await self._qreq.put(req)
        await self._qreq.join()
        self._source_exhausted.set()


class VedaStreamFetcher(BaseVedaSetFetcher):
    def __init__(self, streamer, **kwargs):
        self.streamer = streamer
        super(VedaStreamFetcher, self).__init__(**kwargs)

    async def write_stack(self):
        await asyncio.sleep(0.0)
        while True:
            try:
                dataload = await self._qwrite.get()
                self.streamer._buf.append(dataload)
                async with self._write_lock:
                    self.streamer._q.put_nowait(dataload)
                    if self.streamer.write_vb:
                        await self.loop.run_in_executor(self._write_executor, self.write_fn, dataload)
                self._qreq.task_done()
                self._qwrite.task_done()
            except CancelledError:
                break
        return True


