import os
import inspect
import numpy as np
from functools import partial
from pyveda.config import VedaConfig
from pyveda.io.remote.client import ThreadedAsyncioRunner, HTTPDataClient

def write_v_sample_obj(vb, data):
    """
    Writes a veda sample (label, image, veda ID)
    to a vedabase object.
    """
    label, image, *data = data
    vb._image_array.append(image)
    vb._label_array.append(label)
    write_v_sample_id(vb, *data)
    return data


def write_v_sample_id(vb, data):
    """
    Writes a veda ID to a vedabase index table.
    """
    vid, *data = data
    row = vb._tables.veda_ids.row
    row["veda_id"] = vid
    row.append()
    vb._tables.veda_ids.flush()
    return data


is_iterator = lambda x: isinstance(x, inspect.collections.Iterator)

def is_stream_based_accessor(accessor):
    if type(accessor).__name__ == "BufferedDataStream":
        return True
    if hasattr(accessor, "__v_iotype__"):
        return accessor.__v_iotype__ == "BufferedIOBased"
    return False


def is_file_based_accessor(accessor):
    if type(accessor).__name__ == "H5DataBase":
        return True
    if hasattr(accessor, "__v_iotype__"):
        return accessor.__v_iotype__ == "FileIOBased"
    return False


def ensure_iterator(obj):
    if not obj:
        return obj
    if is_iterator(obj):
        return obj
    if inspect.isgeneratorfunction(obj):
        return obj()
    if inspect.isgenerator(obj):
        gs = inspect.getgeneratorstate(obj)
        if gs is inspect.GEN_CLOSED:
            raise GeneratorExit("Input source generator is empty")
        # Raise warning/do something if already running(suspended)?
        return gs
    if is_iterable(obj):
        return iter(obj)

    return False


def configure_client(vset, source=None, token=None, **kwargs):
    if not token:
        token = VedaConfig().conn.access_token

    source = source or getattr(vset, "_source_factory", None)
    if source:
        source = ensure_iterator(source)
    if not source:
        raise ValueError("why u do that")

    if is_file_based_accessor(accessor):
        write_fn = partial(write_v_sample_obj(vb=vset))
        abf = VedasetFetcher(source,
                             token=token,
                             total_count=vset.count,
                             write_fn=write_fn)



    if is_stream_based_accessor(accessor):
        pass



class IOTaskExecutor(object):
    schedulers = ("synchronous",
                  "concurrent",)

    def __init__(self, client=None, scheduler="concurrent"):
        self.client = client
        if scheduler not in self.schedulers:
            raise NotImplementedError("Scheduler '{}' not supported")

    def _run_sync(self, client):
        client = self.client or client
        assert client
        with ThreadedAsyncIORunner(
            client.run_loop, client.start_fetch) as tar:
            tar(loop=tar._loop)






def build_vedabase(database, source, partition, total, token, label_threads=1, image_threads=10):
    abf = HTTPDataClient(source, total_count=total, token=token,
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


