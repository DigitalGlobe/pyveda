import asyncio
import collections
import queue
import threading
import time
from functools import partial
import numpy as np

from pyveda.io.remote.client import HTTPDataClient
from pyveda.vedaset.base import BaseSampleArray, BaseDataSet
from pyveda.vedaset.interface import BaseVariableArray, ArrayTransformPlugin
from pyveda.frameworks.batch_generator import VedaStreamGenerator
#from pyveda.vv.labelizer import Labelizer


class BufferedVariableArray(ArrayTransformPlugin):
    pass


class BufferedSampleArray(BaseSampleArray):
    def __init__(self, vset, group):
        super(BufferedSampleArray, self).__init__(vset, group)
        self._n_consumed = 0
        self._exhausted = False
        if self.allocated == 0:
            self.exhausted = True

    def __iter__(self):
        return self

    def __next__(self):
        # Order needs to be [image, label]
        while self._n_consumed < self.allocated:
            try:
                nreqs = next(self._vset._gen)
            except StopIteration:
                pass
            else:
                asyncio.run_coroutine_threadsafe(
                    self._vset._fetcher.produce_reqs(reqs=[nreqs]),
                    loop=self._vset._loop)

            # The following get() blocks, as it should, when we're waiting for
            # the thread running the asyncio loop to fetch more data while the
            # source generator is not yet exhausted
            dps = self._vset._q.get()
            label, image = dps
            self._n_consumed += 1
            return [image, label]

        self.exhausted = True
        raise StopIteration

    def batch_iter(self, batch_size):
        while True:
            batch = []
            while len(batch) < batch_size:
                batch.append(self.__next__())
            yield batch

    def batch_generator(self, batch_size, shuffle=True, channels_last=False,
                        rescale=False, flip_horizontal=False, flip_vertical=False,
                        **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaStream partition.
        #Arguments
            batch_size: Int. batch size
            shuffle: Boolean.
            channels_last: Boolean. To return image data as Height-Width-Depth,
            instead of the default Depth-Height-Width
            rescale: boolean. Rescale image values between 0 and 1.
            flip_horizontal: Boolean. Horizontally flip image and lables.
            flip_vertical: Boolean. Vertically flip image and lables
        """
        return VedaStreamGenerator(self, batch_size=batch_size, shuffle=shuffle,
                                   channels_last=channels_last, rescale=rescale,
                                   flip_horizontal=flip_horizontal,
                                   flip_vertical=flip_vertical, **kwargs)

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        self._exhausted = val
        if val:
            self._vset._on_group_exhausted()


    def clean(self, count=None):
        """
        Page through VedaStream data and flag bad data.
        Params:
            count: the number of tiles to clean
        """
        classes = self._vset.classes
        mltype = self._vset.mltype
        Labelizer(self, mltype, count, classes).clean()


class BufferedDataStream(BaseDataSet):
    _sample_class = BufferedSampleArray
    _variable_class = BufferedVariableArray

    def __init__(self, source, bufsize=100, auto_startup=False,
                 auto_shutdown=False, write_index=True, write_h5=False,
                 **kwargs):
        super(BufferedDataStream, self).__init__(**kwargs)
        self._gen = source
        self._auto_startup = auto_startup
        self._auto_shutdown = auto_shutdown
        self._exhausted = False
        self._bufsize = bufsize if bufsize < self.count else self.count

        self._fetcher = None
        self._loop = None
        self._q = queue.Queue()
        self._buf = collections.deque(maxlen=bufsize)
        self._thread = None

        self._write_index = write_index
        self._write_h5 = write_h5

        if auto_startup:
            self._start_consumer()

    @property
    def _image_array(self):
        _, imgs, _ = zip(*self._buf)
        return self._variable_class(self, imgs)

    @property
    def _label_array(self):
        lbls, _, _ = zip(*self._buf)
        return self._variable_class(self, lbls)

    @property
    def exhausted(self):
        return self._exhausted

    @exhausted.setter
    def exhausted(self, val):
        assert isinstance(val, bool)
        if val:
            self._on_exhausted()
        self._exhausted = val

    def _on_group_exhausted(self):
        if all([self.train.exhausted,
                self.test.exhausted,
                self.validate.exhausted]):
            self.exhausted = True

    def _on_exhausted(self):
        if self._auto_shutdown:
            self._stop_consumer()

    def _initialize_buffer(self):
        reqs = []
        while len(reqs) < self._bufsize:
            try:
                reqs.append(next(self._gen))
            except StopIteration:
                break

        f = asyncio.run_coroutine_threadsafe(
            self._fetcher.produce_reqs(reqs=reqs), loop=self._loop)
        f.result()

    def _configure_fetcher(self, **kwargs):
        if self._write_h5:
            vb = VedaBase.from_vtype("temp.h5", self._unpack())
            write_fn = partial(vb_write_fn, vb=vb)
            kwargs.update({"write_fn": write_fn})
            self._vb = vb

        img_h = self._img_handler_class
        lbl_h = self._lbl_handler_class

        self._fetcher = HTTPDataClient(self,
                                       total_count=self.count,
                                       img_payload_handler=img_h,
                                       lbl_payload_handler=lbl_h,
                                       **kwargs)

    def _configure_worker(self, fetcher=None, loop=None):
        if not self._fetcher:
            self._configure_fetcher()
        if not loop:
            loop = asyncio.new_event_loop()
        self._loop = loop
        self._thread = threading.Thread(
            target=partial(self._fetcher.run_loop, loop=loop))

    def _start_consumer(self, init_buff=True):
        if not self._fetcher:
            self._configure_fetcher()
        if not self._thread:
            self._configure_worker()

        self._thread.start()
        time.sleep(1.0)
        self._consumer_fut = asyncio.run_coroutine_threadsafe(
            self._fetcher.start_fetch(self._loop), loop=self._loop)
        if init_buff:
            self._initialize_buffer()

    def _stop_consumer(self):
        self._consumer_fut.cancel()
        f = asyncio.run_coroutine_threadsafe(
            self._fetcher.kill_workers(), loop=self._loop)
        f.result() # Wait for workers to shutdown gracefully
        for task in asyncio.Task.all_tasks():
            task.cancel()
        self._loop.create_task(self._fetcher.session.close())
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    @classmethod
    def from_vc(cls, vc, **kwargs):
        return cls(vc.mltype, vc.classes,
                   vc.count, vc.gen_sample_ids(**kwargs),
                   vc.imshape, **kwargs)

    def __enter__(self):
        self._start_consumer()
        return self

    def __exit__(self, *args):
        self._stop_consumer()

    def __len__(self):
        return self.count


