import asyncio
try:
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
except ImportError:
    pass
import aiohttp
from concurrent.futures import CancelledError, TimeoutError

import numpy as np
from skimage.io import imread

from tempfile import NamedTemporaryFile
import os

import concurrent.futures

#executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
def on_fail(shape=(8, 256, 256), dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

class AsyncBatchFetcher(object):
    def __init__(self, reqs, token, max_retries=5, timeout=20, session=None, session_limit=30,
                 reqs_limit=500, pproc_poolsize=10, trace_configs=[],
                 connector=aiohttp.TCPConnector, executor=concurrent.futures.ThreadPoolExecutor(max_workers=10)):
        self.reqs = reqs
        self.headers = {"Authorization": "Bearer {}".format(token)}
        self.reqs_limit = min(len(reqs), reqs_limit)
        self.max_retries = max_retries
        self.timeout = timeout
        self.session = session
        self.session_limit = session_limit
        self.pproc_poolsize = pproc_poolsize
        self.trace_configs = trace_configs
        self.connector = connector
        self.executor = executor
        self.results = {}

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
        return arr

    async def consume_reqs(self, session):
        await asyncio.sleep(0.0)
        while True:
            try:
                url, index, tries = await self._qreq.get()
                tries += 1
                async with session.get(url) as response:
                    response.raise_for_status()
                    bstring = await response.read()
                    await self._qres.put([index, bstring])
                    self._qreq.task_done()
                    await response.release()
            except CancelledError as ce:
                break
            except Exception as e:
                if tries < self.max_retries:
                    await asyncio.sleep(0.0)
                    await self._qreq.put([url, index, tries])
                    self._qreq.task_done()
                else:
                    await self._qres.put([index, None])
                    self._qreq.task_done()

    async def produce_reqs(self):
        for req in self.reqs:
            await self._qreq.put(req)

    async def process(self, loop):
        while True:
            try:
                index, payload = await self._qres.get()
                if not payload:
                    arr = on_fail()
                else:
                    arr = await loop.run_in_executor(self.executor, self.bytes_to_array, payload)
                self.results[index] = arr
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
        async with aiohttp.ClientSession(loop=loop, connector=self.connector(limit=self.session_limit),
                                        headers=self.headers, trace_configs=self.trace_configs) as session:
            results = await self.fetch(session, loop)
            return results

    def run(self, loop=None):
        if loop:
            do_shutdown = False
        else:
            loop = asyncio.get_event_loop()
            do_shutdown = True
        results = loop.run_until_complete(self.run_fetch(loop))
        loop.run_until_complete(asyncio.sleep(0.250))
        if do_shutdown:
            loop.close()
#        return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="json file list of urls", default="boatset_coll.json")
    parser.add_argument("--num", help="number of urls from file to process", default=None)
    parser.add_argument("--nconn", help="max number of concurrent connections", default=None)
    parser.add_argument("--debug", help="display runtime stats to stdout, ouput profile tracer stats", default=False)
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
        reqs.append([url, tuple(index), 0])
    headers = {"Authorization": "Bearer {}".format(token)}
    abf = AsyncBatchFetcher(reqs, token, trace_configs=trace_configs)
    loop = asyncio.get_event_loop()
#    results = loop.run_until_complete(run_fetch(reqs, nconn, headers, loop, trace_configs=trace_configs))
    abf.run(loop=loop)

    if args.debug:
        stop = timeit.default_timer()
        total_time = stop - start
        mins, secs = divmod(total_time, 60)
        hours, mins = divmod(mins, 60)
        sys.stdout.write("Total running time: %d:%d:%d.\n" % (hours, mins, secs))
        if trace:
            basepath, inputfile = os.path.split(args.file)
            basename = "_".join([inputfile.split(".")[0], "n{}".format(args.num)])
            filename = mklogfilename(basename, suffix="json", path=basepath)
            with open(filename, "w") as f:
                json.dump(trace.cache, f)
            sys.stdout.write("Tracer stats output file written to {}".format(filename))
#    trainer.close()
    loop.close()

