from functools import partial
import os
from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread
from pyveda.utils import extract_load_tasks
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, VedaBaseFetcher

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


def write_data(data, datagroup=None):
    images, labels = data
    datagroup.images.append(images)
    datagroup.labels.append(labels)

def write_fetch(database, source, partition, total, token):
    abf = VedaBaseFetcher(source, total_count=total, token=token, write_fn=partial(write_data, datagroup=database),
                          lbl_payload_handler=database.labels._from_geo, img_payload_handler=bytes_to_array)
    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)


