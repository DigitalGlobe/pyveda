from functools import partial
import os
import numpy as np
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, VedaBaseFetcher

def vedabase_batch_write(data, database=None, partition=[70, 20, 10]):
    trainp, testp, valp = partition
    images, labels = data
    batch_size = labels.shape[0]
    ntrain = round(batch_size * (trainp * 0.01))
    ntest = round(batch_size * (testp * 0.01))
    nval = round(batch_size * (valp * 0.01))

    # write training data
    database.train.images.append(images[:ntrain])
    database.train.labels.append(labels[:ntrain])

    # write testing data
    test_start = ntrain + 1
    test_stop = ntrain + ntest
    database.test.images.append(images[test_start:test_stop])
    database.test.labels.append(labels[test_start:test_stop])

    # write validation data
    val_start = test_stop + 1
    database.validate.images.append(images[val_start:])
    database.validate.labels.append(labels[val_start:])

def build_vedabase(database, source, partition, total, token, label_threads=1, image_threads=10):
    abf = VedaBaseFetcher(source, total_count=total, token=token,
                          write_fn=partial(vedabase_batch_write, database=database, partition=partition),
                          img_payload_handler=database._image_klass.bytes_to_array,
                          lbl_payload_handler=partial(database._label_klass.from_geo, imshape=database.image_shape),
                          num_lbl_payload_threads=label_threads, num_img_payload_threads=image_threads)

    with ThreadedAsyncioRunner(abf.run) as tar:
        tar(loop=tar._loop)


