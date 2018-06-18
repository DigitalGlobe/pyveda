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

def on_fail(shape=(8, 256, 256), dtype=np.float32):
    return np.zeros(shape, dtype=dtype)


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


class AsyncBatchFetcher(BatchFetchTracer):
    def __init__(self, reqs=[], token=None, max_retries=5, timeout=20, session=None,
                 session_limit=30, reqs_limit=500, proc_poolsize=10, connector=aiohttp.TCPConnector,
                 executor=concurrent.futures.ThreadPoolExecutor(max_workers=10), on_result = lambda x: x,
                 run_tracer=False, **kwargs):

        super(AsyncBatchFetcher, self).__init__(**kwargs)
        self.reqs = reqs
        self._token = token
        self.reqs_limit = min(len(reqs), reqs_limit)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session
        self.session_limit = session_limit
        self.proc_poolsize = proc_poolsize
        self._trace_configs = []
        self._connector = connector
        self._executor = executor
        self._run_tracer = run_tracer
        self.on_result = on_result
        if run_tracer:
            trace_config = self._configure_tracer()
            self._trace_configs.append(trace_config)

        self.results = {}

    @property
    def headers(self):
        if not self._token:
            raise AttributeError("Provide an auth token on init or as an argument to run")
        return {"Authorization": "Bearer {}".format(self._token)}

    def bytes_to_array(self, bstring):
        if bstring is None:
            return onfail()
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
            self.on_result(arr)

        return arr

    async def consume_reqs(self, session):
        await asyncio.sleep(0.0)
        while True:
            try:
                url, tries = await self._qreq.get()
                tries += 1
                async with session.get(url) as response:
                    response.raise_for_status()
                    bstring = await response.read()
                    await self._qres.put([url, bstring])
                    self._qreq.task_done()
                    await response.release()
            except CancelledError as ce:
                break
            except Exception as e:
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

    async def process(self, loop):
        while True:
            try:
                url, payload = await self._qres.get()
                if not payload:
                    arr = on_fail()
                else:
                    arr = await loop.run_in_executor(self.executor, self.bytes_to_array, payload)
                self._qres.task_done()
            except CancelledError as ce:
                break
        return True

    async def fetch(self, session, loop):
        self.results = {}
        self._qreq, self._qres = asyncio.Queue(maxsize=self.reqs_limit), asyncio.Queue()
        consumers = [asyncio.ensure_future(self.consume_reqs(session)) for _ in range(self.reqs_limit)]
        producer = await self.produce_reqs()
        processors = [asyncio.ensure_future(self.process(loop)) for _ in range(self.pproc_poolsize)]
        await self._qreq.join()
        await self._qres.join()
        for fut in consumers:
            fut.cancel()
        for fut in processors:
            fut.cancel()
        done, pending = await asyncio.wait(processors)
        return self.results

    async def run_fetch(self, loop):
        async with aiohttp.ClientSession(loop=loop,
                                         connector=self.connector(limit=self.session_limit),
                                         headers=self.headers,
                                         trace_configs=self.trace_configs) as session:
            results = await self.fetch(session, loop)
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

    abf = AsyncBatchFetcher(reqs=reqs, token=token, trace_configs=trace_configs)
    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)

