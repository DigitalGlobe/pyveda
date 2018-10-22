from functools import partial

from pyveda.utils import extract_load_tasks
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, AsyncArrayFetcher

def write_data(data, datagroup=None):
    images, labels = data
    datagroup.images.append(images)
    datagroup.labels.append(labels)

def write_vedadb(data, db, partition):
    images, labels = data
    pass

def write_fetch(points, datagroup):
    reqs = []
    lut = {}
    for p in points:
        url, token = extract_load_tasks(p.image.dask)
        reqs.append([url])
        lut[url] = p.y

    abf = AsyncArrayFetcher(reqs=reqs, token=token, label_lut=lut, write_fn=partial(write_data, datagroup=datagroup))
    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)


