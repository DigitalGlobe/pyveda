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
import inspect

from pyveda.io.utils import write_trace_profile
from pyveda.config import VedaConfig
from pyveda.utils import tail

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
        self._ibuf = collections.deque(maxlen=maxlen)

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
                 batch_size=100,
                 suppress_callback_errors = True,
                 suppress_http_errors = True,
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
        self._io_executor = write_executor(max_workers=num_write_threads)
        self._lbl_payload_executor = lbl_payload_executor(max_workers=num_lbl_payload_threads)
        self._img_payload_executor = img_payload_executor(max_workers=num_img_payload_threads)


        self.lbl_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._lbl_payload_executor,
                                                     fn=lbl_payload_handler)
        self.img_payload_handler = functools.partial(self._payload_handler,
                                                     executor=self._img_payload_executor,
                                                     fn=img_payload_handler)
        if not buf:
            batch_size = min(batch_size, max_memarrays)
            buf = PhaseBuffer(period=batch_size, maxlen=max_memarrays)
        self.data_buf = buf

        self.io_callbacks = []
        self.cpu_callbacks = []
        self._logger = logger

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
        self._data_lock = asyncio.Lock(loop=loop)
        self._consumers = [asyncio.ensure_future(self.consume_reqs(), loop=loop)
                           for _ in range(self.max_concurrent_reqs)]
        self._writers = [asyncio.ensure_future(self.process_data(), loop=loop)
                         for _ in range(self._n_write_workers)]

    def register_io_callback(self, fn, executor=None, use_lock=False, lock=None):
        if not executor:
            executor = self._io_executor
        if use_lock:
            if not lock:
                lock = asyncio.Lock(loop=self.loop)

        def wrap(fn):
            if lock:
                async def wrapped(*args, **kwargs):
                    async with lock:
                        return await self.loop.run_in_executor(executor, fn, *args)
            else:
                async def wrapped(*args, **kwargs):
                    return await self.loop.run_in_executor(executor, fn, *args)
            return wrapped

        self.io_callbacks.append(wrap(fn))

    def register_cpu_callback(self, fn, executor=None, use_lock=False, lock=None):
        if not executor:
            executor = self._cpu_executor
        if use_lock:
            if not lock:
                lock = asyncio.Lock(loop=self.loop)

        def wrap(fn):
            if lock:
                async def wrapped(*args, **kwargs):
                    async with lock:
                        return await self.loop.run_in_executor(executor, fn, *args)
            else:
                async def wrapped(*args, **kwargs):
                    return await self.loop.run_in_executor(executor, fn, *args)
            return wrapped

        self.cpu_callbacks.append(wrap(fn))

    async def kill_workers(self):
        await self._qreq.join()
        await self._qwrite.join()
        # Next line runs io on any remaining data in the buffer
        if not self.data_buf.phased:
            await self.run_io_callbacks(self.data_buf.current_sample)
        for fut in self._consumers:
            fut.cancel()
        for fut in self._writers:
            fut.cancel()
        done, pending = await asyncio.wait(self._writers)
        logger.info("Shutdown successful")
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

    async def run_io_callbacks(self, dstack):
        for cb in self.io_callbacks:
            try:
                await cb(dstack)
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


class HTTPVedaClient(HTTPDataClient):
    def __init__(self, **kwargs):
        super(HTTPVedaClient, self).__init__(**kwargs)
        if self._token is None:
            self._token = cfg.conn.access_token
