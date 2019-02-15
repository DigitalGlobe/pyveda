import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
import dask

threads = int(os.environ.get('GBDX_THREADS', 64))
pool = ThreadPoolExecutor(threads)
threaded_get = partial(dask.threaded.get, num_workers=threads)


def url_to_array(dsk):
    return(dsk.compute())


def write_fetch(points, labelgroup, datagroup):
    futures = []
    for p in points:
        futures.append(pool.submit(url_to_array, p.image))
        labelgroup.append(p.y)

    for f in as_completed(futures):
        datagroup.image.append(f.result())  # This should be sync
