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

MAX_RETRIES = 5
MAX_CONNECTIONS = 100
TIMEOUT = 20

def on_fail(shape=(8, 256, 256), dtype=np.float32):
    return np.zeros(shape, dtype=dtype)

def bytes_to_array(bstring):
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

async def consume_reqs(qreq, qres, session, max_tries=5):
    await asyncio.sleep(0.1)
    while True:
        try:
            url, index, tries = await qreq.get()
            tries += 1
            async with session.get(url) as response:
                response.raise_for_status()
                bstring = await response.read()
                await qres.put([index, bstring])
                qreq.task_done()
                await response.release()
        except CancelledError as ce:
            break
        except Exception as e:
            print(e)
            if tries < max_tries:
                await asyncio.sleep(0.1)
                await qreq.put([url, index, tries])
                qreq.task_done()
            else:
                await qres.put([index, None])
                qreq.task_done()

async def produce_reqs(qreq, reqs):
    for req in reqs:
        await qreq.put(req)

async def process(qres, results):
    while True:
        try:
            index, payload = await qres.get()
            if not payload:
                arr = on_fail()
                arr = False
            else:
                arr = await loop.run_in_executor(None, bytes_to_array, payload)
            results[index] = arr
            qres.task_done()
        except CancelledError as ce:
            break
    return True

async def fetch(reqs, session, nconn, batch_size=2000, nprocs=5):
    results = {}
    qreq, qres = asyncio.Queue(maxsize=batch_size), asyncio.Queue()
    consumers = [asyncio.ensure_future(consume_reqs(qreq, qres, session)) for _ in range(nconn)]
    producer = await produce_reqs(qreq, reqs)
    processors = [asyncio.ensure_future(process(qres, results)) for _ in range(nprocs)]
    await qreq.join()
    await qres.join()
    for fut in consumers:
        fut.cancel()
    for fut in processors:
        fut.cancel()
    done, pending = await asyncio.wait(processors)
    return results

async def run_fetch(reqs, nconn, headers, loop, trace_configs=[]):
    async with aiohttp.ClientSession(loop=loop, connector=aiohttp.TCPConnector(limit=nconn),
                                     headers=headers, trace_configs=trace_configs) as session:
        results = await fetch(reqs, session, nconn)
        return results


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
    loop = asyncio.get_event_loop()

    results = loop.run_until_complete(run_fetch(reqs, nconn, headers, loop, trace_configs=trace_configs))
    loop.run_until_complete(asyncio.sleep(0))

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
    loop.close()

