from functools import partial
import os
from tempfile import NamedTemporaryFile
import numpy as np
from skimage.io import imread
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, VedaBaseFetcher

def vedabase_batch_write(data, database=None, partition=[70, 20, 10]):
    trainp, testp, valp = partition
    batch_size = data.shape[0]
    ntrain = round(batch_size * (trainp * 0.01))
    ntest = round(batch_size * (testp * 0.01))
    nval = round(batch_size * (valp * 0.01))
    images, labels = data

    # write training data
    database.train.images.append(images[:ntrain]))
    database.labels.append(labels[:ntrain]))
    # write testing data
    test_start = ntrain + 1
    test_stop = ntrain + ntest
    database.test.images.append(images[test_start:test_stop])
    database.test.labels.append(labels[test_start:test_stop])
    # write validation data
    val_start = test_stop + 1
    database.validation.images.append(images[val_start:])
    database.validation.labels.append(labels[val_start:])

def build_vedabase(database, source, partition, total, token):
    abf = VedaBaseFetcher(source, total_count=total, token=token,
                          write_fn=partial(vedabase_batch_write, database=database, partition=partition),
                          lbl_payload_handler=partial(database._label_klass.from_geo, imshape=database.image_shape),
                          img_payload_handler=database._image_klass.bytes_to_array)

    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)


