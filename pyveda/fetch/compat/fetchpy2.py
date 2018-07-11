import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask

threads = int(os.environ.get('GBDX_THREADS', 64))
pool = ThreadPoolExecutor(threads)
threaded_get = partial(dask.threaded.get, num_workers=threads)

def write_fetch(points, labelgroup, datagroup):
    def group_append(dsk):
        datagroup.image.append(dsk.compute())

    futures = []
    for p in points:
        futures.append(pool.submit(group_append, p.image))
        labelgroup.append(p.y)

    finished = []
    for f in as_completed(futures):
        finished.append(f.result())


