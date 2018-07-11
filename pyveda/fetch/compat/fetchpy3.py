from pyveda.utils import extract_load_tasks
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, AsyncBatchFetcher

def write_fetch(points, labelgroup, datagroup):
    reqs = []
    for p in points:
        url, token = extract_load_tasks(p.image.dask)
        reqs.append([url])
        labelgroup.append(p.y)

    abf = AsyncBatchFetcher(reqs=reqs, token=token, on_result=datagroup.image.append)
    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)


