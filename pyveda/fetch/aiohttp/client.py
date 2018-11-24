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
import logging

from pyveda.fetch.diagnostics import BatchFetchTracer
from pyveda.utils import write_trace_profile

has_tqdm = False
try:
    from tqdm import trange, tqdm, tqdm_notebook, tnrange
    has_tqdm = True
except ImportError:
    pass

logger = logging.getLogger(__name__)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.setLevel(logging.INFO)
fh = logging.FileHandler('fetcher.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)


class ThreadedAsyncioRunner(object):
    def __init__(self, method, loop=None):
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._method = method
        self._thread = threading.Thread(target=self._loop.run_forever)
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


class AsyncBaseFetcher(BatchFetchTracer):
    def __init__(self, reqs=[], token=None, payload_handler=lambda x: x, max_retries=5,
                 timeout=20, session=None, session_limit=30, max_concurrent_requests=500,
                 num_payload_workers=10, num_payload_threads=None, connector=aiohttp.TCPConnector,
                 run_tracer=False, **kwargs):

        super(AsyncBaseFetcher, self).__init__(**kwargs)
        self.reqs = reqs
        self._token = token
        self.max_concurrent_reqs = min(len(reqs), max_concurrent_requests)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session
        self.session_limit = session_limit
        self._payload_handler = payload_handler
        self._n_payload_workers = num_payload_workers
        if not num_payload_threads:
            num_payload_threads = num_payload_workers
        self._n_payload_threads = num_payload_threads
        self._payload_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_payload_threads)
        self._connector = connector
        self._trace_configs = []
        self._connector = connector
        self._run_tracer = run_tracer
        if run_tracer:
            trace_config = self._configure_tracer()
            self._trace_configs.append(trace_config)

        self.results = {}

    @property
    def headers(self):
        if not self._token:
            raise AttributeError("Provide an auth token on init or as an argument to run")
        return {"Authorization": "Bearer {}".format(self._token)}

    async def on_result(self, *args, **kwargs):
        await asyncio.sleep(0.0)

    async def consume_reqs(self, session):
        await asyncio.sleep(0.0)
        while True:
            try:
                url, tries = await self._qreq.get()
                tries += 1
                async with session.get(url) as response:
                    response.raise_for_status()
                    bstring = await response.read()
                    logger.info("  URL READ SUCCESS: {}".format(url))
                    await self._qres.put([url, bstring])
                    self._qreq.task_done()
                    await response.release()
            except CancelledError as ce:
                break
            except Exception as e:
                logger.info("    URL READ FAIL: {}".format(url))
                if tries < self.max_retries:
                    await asyncio.sleep(0.0)
                    await self._qreq.put([url, tries])
                    self._qreq.task_done()
                else:
                    await self._qres.put([url, None])
                    self._qreq.task_done()

    async def produce_reqs(self):
        for req in self.reqs:
            await self._qreq.put((req[0], 0))

    async def handle_payload(self, loop):
        while True:
            try:
                url, payload = await self._qres.get()
                if not payload:
                    arr = on_fail()
                else:
                    arr = await loop.run_in_executor(self._payload_executor, self._payload_handler, payload)
                await self.on_result([url, arr])
                self._qres.task_done()
            except CancelledError as ce:
                break
        return True

    def _configure(self, session, loop):
        self.results = {}
        self._qreq = asyncio.Queue(maxsize=self.max_concurrent_reqs)
        self._qres = asyncio.Queue()
        self._consumers = [asyncio.ensure_future(self.consume_reqs(session)) for _ in range(self.max_concurrent_reqs)]
        self._handlers = [asyncio.ensure_future(self.handle_payload(loop)) for _ in range(self._n_payload_workers)]

    async def fetch(self, session, loop):
        self._configure(session, loop)
        producer = await self.produce_reqs()
        await self._qreq.join()
        await self._qres.join()
        for fut in self._consumers:
            fut.cancel()
        for fut in self._handlers:
            fut.cancel()
        done, pending = asyncio.wait(self._handlers)
        return self.results

    async def run_fetch(self, loop):
        async with aiohttp.ClientSession(loop=loop,
                                         connector=self._connector(limit=self.session_limit),
                                         headers=self.headers,
                                         trace_configs=self._trace_configs) as session:
            logger.info("BATCH FETCH START")
            results = await self.fetch(session, loop)
            logger.info("BATCH FETCH COMPLETE")
            return results

    async def run(self, reqs=None, token=None, loop=None):
        if not loop:
            loop = asyncio.get_event_loop()
        if reqs:
            self.reqs = reqs
        else:
            assert(len(self.reqs) > 0)
        if token:
            self._token = token

        fut = await self.run_fetch(loop)
        await asyncio.sleep(0.250)
        return fut


class AsyncArrayFetcher(AsyncBaseFetcher):
    def __init__(self, write_fn=lambda x: x, payload_handler=lambda x: x, max_memarrays=200,
                 num_write_workers=1, num_write_threads=1, label_lut={}, *args, **kwargs):

        super(AsyncArrayFetcher, self).__init__(payload_handler=payload_handler, *args, **kwargs)
        self.write_fn = write_fn
        self._n_write_workers = num_write_workers
        self._n_write_threads = num_write_threads
        self._label_lut = label_lut
        self._max_memarrs = int(np.floor(max_memarrays / float(num_write_workers)))
        self._write_executor = concurrent.futures.ThreadPoolExecutor(max_workers=num_write_threads)
        if has_tqdm:
            self._pbar = tqdm(total=len(self.reqs))

    async def on_result(self, res):
        url, arr = res
        label = self._label_lut[url]
        await self._qwrite.put([arr, label])

    async def write_stack(self, loop):
        arrs, labels = [], []
        while True:
            try:
                arr, label = await self._qwrite.get()
                arrs.append(arr)
                labels.append(label)
                if len(arrs) == self._max_memarrs:
                    data = [np.array(arrs), np.array(labels)]
                    async with self._write_lock:
                        await loop.run_in_executor(self._write_executor, self.write_fn, data)
                    arrs, labels = [], []
                self._qwrite.task_done()
                if has_tqdm:
                    self._pbar.update(1)
            except CancelledError as ce: # Write out anything remaining
                if len(arrs) > 0:
                    data = [np.array(arrs), np.array(labels)]
                    async with self._write_lock:
                        await loop.run_in_executor(self._write_executor, self.write_fn, data)
                break
        return True

    def _configure(self, session, loop):
        super(AsyncArrayFetcher, self)._configure(session, loop)
        self._write_lock = asyncio.Lock()
        self._qwrite = asyncio.Queue()
        self._writers = [asyncio.ensure_future(self.write_stack(loop)) for _ in range(self._n_write_workers)]

    async def fetch(self, session, loop):
        self._configure(session, loop)
        producer = await self.produce_reqs()
        await self._qreq.join()
        await self._qres.join()
        await self._qwrite.join()
        for fut in self._consumers:
            fut.cancel()
        for fut in self._handlers:
            fut.cancel()
        for fut in self._writers:
            fut.cancel()
        done, pending = await asyncio.wait(self._writers)
        return self.results



class VedaBaseFetcher(BatchFetchTracer):
    def __init__(self, reqs, total_count=0, session=None, token=None, max_retries=5, timeout=20,
                 session_limit=30, max_concurrent_requests=200, max_memarrays=100, connector=aiohttp.TCPConnector,
                 lbl_payload_handler=lambda x: x, img_payload_handler=lambda x: x, write_fn=lambda x: x,
                 img_batch_transform=lambda x: x, lbl_batch_transform=lambda x: x,
                 num_lbl_payload_threads=1, num_img_payload_threads=10, num_write_workers=1, num_write_threads=1,
                 lbl_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 img_payload_executor=concurrent.futures.ThreadPoolExecutor,
                 write_executor=concurrent.futures.ThreadPoolExecutor,
                 run_tracer=False, *args, **kwargs):

        self.reqs = reqs
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
        self.max_memarrs = max_memarrays
        self._lbl_batch_transform = lbl_batch_transform
        self._img_batch_transform = img_batch_transform
        self._n_lbl_payload_threads = num_lbl_payload_threads
        self._n_img_payload_threads = num_img_payload_threads
        self._lbl_payload_executor = lbl_payload_executor(max_workers=num_lbl_payload_threads)
        self._img_payload_executor = img_payload_executor(max_workers=num_img_payload_threads)
        self._n_write_workers = num_write_workers
        self._n_write_threads = num_write_threads
        self._write_executor = write_executor(max_workers=num_write_threads)
        self._pbar = None
        if has_tqdm and total_count:
            self._pbar = tqdm(total=total_count)

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
                return data
            except CancelledError:
                break
            except Exception as e:
                logger.info(e)
                logger.info("    URL READ ERROR: {}".format(url))
                retries -= 1
        return None

    async def write_stack(self):
        await asyncio.sleep(0.0)
        labels, images = [], []
        while True:
            try:
                label, image = await self._qwrite.get()
                labels.append(label)
                images.append(image)
                if len(images) == self.max_memarrs:
                    data = [self._img_batch_transform(images), self._lbl_batch_transform(labels)]
                    async with self._write_lock:
                        try:
                            await self.loop.run_in_executor(self._write_executor, self.write_fn, data)
                            logger.info("SUCCESS WRITE {} DATAPOINTS".format(len(images)))
                        except Exception as e:
                            logger.info("Exception is WRITE_STACK: {}".format(e))
                    labels, images = [], []
                self._qreq.task_done()
                self._qwrite.task_done()
                if self._pbar:
                    self._pbar.update(1)
            except CancelledError: # write out anything remaining
                if images:
                    data = [self._img_batch_transform(images), self._lbl_batch_transform(labels)]
                    async with self._write_lock:
                        await self.loop.run_in_executor(self._write_executor, self.write_fn, data)
                break
        return True

    async def consume_reqs(self):
        while True:
            try:
                label_url, image_url = await self._qreq.get()
                flbl = asyncio.ensure_future(self.fetch_with_retries(label_url, callback=self.lbl_payload_handler))
                fimg = asyncio.ensure_future(self.fetch_with_retries(image_url, json=False, callback=self.img_payload_handler))
                label, image = await asyncio.gather(flbl,fimg)
                await self._qwrite.put([label, image])
            except CancelledError:
                break

    async def produce_reqs(self):
        for req in self.reqs:
            await self._qreq.put(req)
        await self._qreq.join()
        self._source_exhausted.set()

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

    async def run(self, reqs=None, token=None, loop=None):
        if not loop:
            loop = asyncio.get_event_loop()
        if reqs:
            self.reqs = reqs
        if token:
            self._token = token
        fut = await self.start_fetch(loop)
        await asyncio.sleep(0.250)
        return fut


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



