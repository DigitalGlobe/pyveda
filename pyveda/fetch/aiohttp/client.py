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
from skimage.io import imread

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


def on_fail(shape=(8, 256, 256), dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

def bytes_to_array(bstring):
    if bstring is None:
        return on_fail()
    try:
        fd = NamedTemporaryFile(prefix='gbdxtools', suffix='.tif', delete=False)
        fd.file.write(bstring)
        fd.file.flush()
        fd.close()
        arr = imread(fd.name)
        if len(arr.shape) == 3:
            arr = np.rollaxis(arr, 2, 0)
        else:
            arr = np.expand_dims(arr, axis=0)
    except Exception as e:
        arr = on_fail()
    finally:
        fd.close()
        os.remove(fd.name)

    return arr


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
    def __init__(self, write_fn=lambda x: x, payload_handler=bytes_to_array, max_memarrays=200,
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


class VedaBaseFetcher(AsyncArrayFetcher):
    pass

class AsyncPayloadMapper(object):
    def __init__(self, q, label=None, image=None):
        self.q = q
        self._label = label
        self._image = image
        self._pushed = False

    @property
    def label(self):
        return self._label

    @label.setter
    async def label(self, label):
        if label and self.image:
            await self.put([label, self.image])
        else:
            self._label = label

    @property
    def image(self):
        return self._image

    @image.setter
    async def image(self, image):
        if image and self.label:
            await self.put([self.label, image])
        else:
            self._image = image

    async def put(self, payload):
        await self.q.put(payload)
        self._pushed = True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="json file list of urls", default="boatset_coll.json")
    parser.add_argument("--num", help="number of urls from file to process", default=None)
    parser.add_argument("--nconn", help="max number of concurrent connections", default=None)
    parser.add_argument("--run-trace", help="run and ouput request tracer profile stats", default=False)
    parser.add_argument("--verbose", help="output info log", default=False)
    parser.add_argument("--debug", help="output debug log", default=False)
    args = parser.parse_args()

    trace_configs = []
    if args.debug:
        import sys
        import timeit
        start = timeit.default_timer()

        import collections
        from pyveda.utils import mklogfilename
        try:
            from pyveda.fetch.diagnostics.aiohttp_tracer import batch_fetch_tracer
            trace, trace_config = batch_fetch_tracer()
            trace_configs.append(trace_config)
        except ImportError:
            trace = None

    import json
    with open(args.file) as f:
        coll = json.load(f)

    if args.num:
        coll = coll[:int(args.num)]
    if args.nconn:
        nconn = int(args.nconn)
    else:
        nconn = min(len(coll), 1000)

    reqs = []
    for url, token, index in coll:
        reqs.append([url])

    abf = AsyncBaseFetcher(reqs=reqs, token=token, trace_configs=trace_configs)
    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)

