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
import logging.handlers

from pyveda.fetch.diagnostics import BatchFetchTracer
from pyveda.utils import write_trace_profile
from pyveda.config import VedaConfig

has_tqdm = False
try:
    from tqdm import trange, tqdm, tqdm_notebook, tnrange
    has_tqdm = True
except ImportError:
    pass

log_fh = ".{}.log".format(__name__)
logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
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
        self._thread = threading.Thread(
            target=functools.partial(
                run_method, loop=loop))
        self._thread.start()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def __call__(self, *args, **kwargs):
        f = asyncio.run_coroutine_threadsafe(
            self._method(*args, **kwargs), self._loop)
        return f.result()


class BaseVedaSetFetcher(BatchFetchTracer):
    def __init__(
            self,
            total_count=0,
            session=None,
            token=None,
            max_retries=5,
            timeout=20,
            session_limit=30,
            max_concurrent_requests=200,
            max_memarrays=100,
            connector=aiohttp.TCPConnector,
            lbl_payload_handler=lambda x: x,
            img_payload_handler=lambda x: x,
            write_fn=lambda x: x,
            img_batch_transform=lambda x: x,
            lbl_batch_transform=lambda x: x,
            num_lbl_payload_threads=1,
            num_img_payload_threads=10,
            num_write_workers=1,
            num_write_threads=1,
            lbl_payload_executor=concurrent.futures.ThreadPoolExecutor,
            img_payload_executor=concurrent.futures.ThreadPoolExecutor,
            write_executor=concurrent.futures.ThreadPoolExecutor,
            run_tracer=False,
            *args,
            **kwargs):

        self.max_concurrent_reqs = min(total_count, max_concurrent_requests)
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

        self.write_fn = write_fn
        self.max_memarrs = max_memarrays
        self._lbl_batch_transform = lbl_batch_transform
        self._img_batch_transform = img_batch_transform
        self._n_lbl_payload_threads = num_lbl_payload_threads
        self._n_img_payload_threads = num_img_payload_threads
        self._lbl_payload_executor = lbl_payload_executor(
            max_workers=num_lbl_payload_threads)
        self._img_payload_executor = img_payload_executor(
            max_workers=num_img_payload_threads)
        self._n_write_workers = num_write_workers
        self._n_write_threads = num_write_threads
        self._write_executor = write_executor(max_workers=num_write_threads)

        self.lbl_payload_handler = functools.partial(
            self._payload_handler,
            executor=self._lbl_payload_executor,
            fn=lbl_payload_handler)
        self.img_payload_handler = functools.partial(
            self._payload_handler,
            executor=self._img_payload_executor,
            fn=img_payload_handler)

    @property
    def headers(self):
        if not self._token:
            self._token = cfg.conn.access_token
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
                label_url, image_url = await self._qreq.get()
                flbl = asyncio.ensure_future(
                    self.fetch_with_retries(
                        label_url, callback=self.lbl_payload_handler))
                fimg = asyncio.ensure_future(
                    self.fetch_with_retries(
                        image_url,
                        json=False,
                        callback=self.img_payload_handler))
                label, image = await asyncio.gather(flbl, fimg)
                await self._qwrite.put([label, image])
            except CancelledError:
                break

    def _configure(self, session, loop):
        self.session = session
        self.loop = loop
        self._source_exhausted = asyncio.Event(loop=loop)
        self._qreq = asyncio.Queue(maxsize=self.max_concurrent_reqs, loop=loop)
        self._qwrite = asyncio.Queue(loop=loop)
        self._write_lock = asyncio.Lock(loop=loop)
        self._consumers = [
            asyncio.ensure_future(
                self.consume_reqs(),
                loop=loop) for _ in range(
                self.max_concurrent_reqs)]
        self._writers = [
            asyncio.ensure_future(
                self.write_stack(),
                loop=loop) for _ in range(
                self._n_write_workers)]

    async def kill_workers(self):
        await self._qwrite.join()
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

    async def write_stack(self):
        raise NotImplementedError

    async def drive_fetch(self, session, loop):
        raise NotImplementedError

    async def produce_reqs(self):
        raise NotImplementedError


class VedaBaseFetcher(BaseVedaSetFetcher):
    def __init__(self, reqs, **kwargs):
        self.reqs = reqs
        super(VedaBaseFetcher, self).__init__(**kwargs)
        self._pbar = None
        if has_tqdm and self._total_count:
            self._pbar = tqdm(total=self._total_count)

    async def write_stack(self):
        await asyncio.sleep(0.0)
        labels, images = [], []
        while True:
            try:
                label, image = await self._qwrite.get()
                labels.append(label)
                images.append(image)
                if len(images) == self.max_memarrs:
                    data = [
                        self._img_batch_transform(images),
                        self._lbl_batch_transform(labels)]
                    async with self._write_lock:
                        try:
                            await self.loop.run_in_executor(self._write_executor, self.write_fn, data)
                            logger.info(
                                "SUCCESS WRITE {} DATAPOINTS".format(
                                    len(images)))
                        except Exception as e:
                            logger.info(
                                "Exception is WRITE_STACK: {}".format(e))
                    labels, images = [], []
                self._qreq.task_done()
                self._qwrite.task_done()
                if self._pbar:
                    self._pbar.update(1)
            except CancelledError:  # write out anything remaining
                if images:
                    data = [
                        self._img_batch_transform(images),
                        self._lbl_batch_transform(labels)]
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
                item = await self._qwrite.get()
                self.streamer._buf.append(item)
                async with self._write_lock:
                    self.streamer._q.put_nowait(item)
                self._qreq.task_done()
                self._qwrite.task_done()
            except CancelledError:
                break
        return True

    async def drive_fetch(self, session, loop):
        self._configure(session, loop)
        await self._source_exhausted.wait()
        res = await self.kill_workers()

    async def produce_reqs(self, reqs=None):
        for req in reqs:
            if req is None:
                self._source_exhausted.set()
                return True
            await self._qreq.put(req)
        return True
