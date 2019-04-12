import os
import inspect
import asyncio
import numpy as np
from functools import partial
from pyveda.config import VedaConfig
from pyveda.io.remote.handlers import set_handlers
from pyveda.io.remote.client import ThreadedAsyncioRunner, HTTPVedaClient

def write_v_sample_obj(vb, data, write_meta=True):
    """
    Writes a veda sample (label, image, veda ID)
    to a vedabase object.
    """
    labels, images, meta, *data = zip(*data)
    vb.train.images.append(np.array(images))
    vb.train.labels.append(np.array(labels))
    if write_meta:
        write_v_sample_meta(vb, meta)
    return data

def write_v_sample_meta(vb, meta):
    """
    Writes a veda ID to a vedabase index table.
    """
    row = vb.metadata.row
    for vid, *m in meta:
        row["vid"] = vid
        row.append()
    vb.metadata.flush()

is_iterator = lambda x: isinstance(x, inspect.collections.Iterator)

def is_stream_based_accessor(accessor):
    if type(accessor).__name__ == "BufferedDataStream":
        return True
    if hasattr(accessor, "__v_iotype__"):
        return accessor.__v_iotype__ == "BufferedIOBased"
    return False


def is_file_based_accessor(accessor):
    if type(accessor).__name__ in ("H5DataBase", "VedaBase",):
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


def configure_client(vset,
                     on_data=lambda x: x,
                     filters=[],
                     io_callbacks=[],
                     loop=None,
                     **kwargs):

    img_h, lbl_h = set_handlers(vset)

    if is_file_based_accessor(vset):
        cb = partial(write_v_sample_obj, vset)
        io_callbacks.append(cb)

    if is_stream_based_accessor(vset):
        on_data = vset._q.put_nowait

    client = HTTPVedaClient(img_payload_handler=img_h,
                            lbl_payload_handler=lbl_h,
                            on_data=on_data,
                            **kwargs)

    if not loop:
        loop = asyncio.new_event_loop()
    client.set_loop(loop)

    for f in filters:
        client.register_filter(f)
    for f in io_callbacks:
        client.register_io_callback(f, use_lock=True)

    return client



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


def build_vedabase(vbase, source, **kwargs):
    client = configure_client(vbase, **kwargs)

    with ThreadedAsyncioRunner(client.run_loop,
                               client.start_fetch,
                               loop=client.loop) as tar:
        tar(reqs=source, shutdown=True)

    return vbase


