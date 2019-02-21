from functools import partial
import os
import numpy as np
from pyveda.fetch.aiohttp.client import ThreadedAsyncioRunner, VedaBaseFetcher

def vb_write_fn(vb, data):
    label, image, _id = data
    vb._img_arr.append(image)
    vb._lbl_arr.append(label)
    idx_row = vb._fileh.root.sample_index.row
    idx_row['idx_cols'] = _id
    idx_row.append()
    vb._fileh.root.sample_index.flush()


def vedabase_batch_write(data, database=None, partition=[70, 20, 10]):
    trainp, testp, valp = partition
    images, labels, ids = data
    batch_size = images.shape[0]
    ntrain = round(batch_size * (trainp * 0.01))
    ntest = round(batch_size * (testp * 0.01))
    nval = round(batch_size * (valp * 0.01))

    # write training data, record ids
    database.train.images.append_batch(images[:ntrain])
    database.train.labels.append_batch(labels[:ntrain])
    database.train.id_table.append(ids[:ntrain])
    database.train.id_table.flush()

    # write testing data, record ids
    database.test.images.append_batch(images[ntrain:ntrain + ntest])
    database.test.labels.append_batch(labels[ntrain:ntrain + ntest])
    database.test.id_table.append(ids[ntrain:ntrain + ntest])
    database.test.id_table.flush()

    # write validation data, record ids
    database.validate.images.append_batch(images[ntrain + ntest:])
    database.validate.labels.append_batch(labels[ntrain + ntest:])
    database.validate.id_table.append(ids[ntrain + ntest:])
    database.validate.id_table.flush()


def build_vedabase(database, source, partition, total, token, label_threads=1, image_threads=10):
    abf = VedaBaseFetcher(source, total_count=total, token=token,
                          write_fn=partial(vedabase_batch_write, database=database, partition=partition),
                          img_batch_transform=database._image_klass._batch_transform,
                          lbl_batch_transform=database._label_klass._batch_transform,
                          img_payload_handler=database._image_klass._payload_handler,
                          lbl_payload_handler=partial(database._label_klass._payload_handler,
                                                      klasses=database.classes,
                                                      out_shape=database.image_shape),
                          num_lbl_payload_threads=label_threads, num_img_payload_threads=image_threads)

    with ThreadedAsyncioRunner(abf.run_loop, abf.start_fetch) as tar:
        tar(loop=tar._loop)


