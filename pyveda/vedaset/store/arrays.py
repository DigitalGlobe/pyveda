import os
from collections import OrderedDict, defaultdict
import numpy as np
import tables
import ujson as json
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.base import BaseVariableArray


class VirtualSubArray(BaseVariableArray):
    """
    This wraps a pytables array with access determined
    by a contiguous index range given by two integers
    """
    def __init__(self, vset, arr):
        super(VirtualSubArray, self).__init__(vset, arr)
        self._start_ = None
        self._stop_ = None
        self._itr = None

    @property
    def _start(self):
        if self._start_ is None:
            self._start_, self._stop_ = self._vset._update_vindex(self)
        return self._start_

    @property
    def _stop(self):
        if self._stop_ is None:
            self._start_, self._stop_ = self._vset._update_vindex(self)
        return self._stop_

    def __getitem__(self, key):
        if isinstance(key, int):
            spec = self._translate_idx(key)
        if isinstance(key, slice):
            spec = self._translate_slice(key)
        return self._arr.__getitem__(spec)

    def __iter__(self):
        return self # return underlying iter?

    def __next__(self):
        # The subtleties of the following line are important to understand:
        # pytables Arrays return themselves in iter methods.
        # The lib implementation of this effectively means that expected iter
        # objs are the same obj, a singleton. That means usage of simulaneous
        # multiple iterators on single array can result in unexpected behavior
        # since there is always only one maintained instance of iter state. See
        # issue https://github.com/PyTables/PyTables/issues/293
        self._itr = self._arr.iterrows(self._start, self._stop)
        return self._itr.__next__()

    def _translate_idx(self, vidx):
        if vidx is None: # Bounce back None slice parts
            return vidx
        idx = vidx + self._start
        if idx > self._stop:
            raise IndexError("Index out of data range")
        return idx

    def _translate_slice(self, sli):
        start, stop, step = sli.start, sli.stop, sli.step
        start = self._translate_idx(start)
        stop = self._translate_idx(stop)
        # None means default to edges
        if start is None:
            start = self._start
        if stop is None:
            stop = self._stop
        return slice(start, stop, step)

    def append_batch(self, items):
        self.append(items)


class NDImageArray(VirtualSubArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 4:
            return item # for batch append stacked arrays
        if len(dims) in (2, 3): # extend single image array along axis 0
            return item.reshape(1, *dims)
        return item # what could this thing be, let it fail

    def create_array(self, group):
        atom = _atom_from_dtype(self._vset.dtype)
        shape = list(self._vset.image_shape)
        shape.insert(0,0)
        shape = tuple(shape)
        self._vset._fileh.create_earray(group, "images", atom=atom, shape=shape))



class LabelArray(VirtualSubArray):
    pass


class ClassificationArray(LabelArray):
    _default_dtype = np.uint8

    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 2:
            return item # for batch append stacked arrays
        return item.reshape(1, *dims)

    def create_array(self, group):
        atom = tables.UInt8Atom()
        shape = (0, len(self._vset.classes))
        self._vset._fileh.create_earray(group, "labels", atom=atom, shape=shape)


class SegmentationArray(LabelArray):
    _default_dtype = np.float32

    def create_array(self, group):
        atom = _atom_from_dtype(self._vset.dtype)
        shape = tuple([s if idx > 0 else 0 for
                       idx, s in enumerate(self._vset.image_shape)])
        self._vset._fileh.create_earray(group, "labels", atom=atom, shape=shape)


class ObjDetectionArray(LabelArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        assert isinstance(item, list)
        return np.fromstring(json.dumps(item), dtype=np.uint8)

    def _output_fn(self, item):
        return json.loads(item.tostring())

    def append_batch(self, items):
        for item in items:
            self.append(item)

    def create_array(self, group, filters=tables.Filters(complevel=0)):
        atom = tables.UInt8Atom()
        self._vset._fileh.create_vlarray(group, "labels", atom=atom, filters=filters)


def get_array_handler(inst):
    if inst.mltype == "classification":
        return ClassificationArray
    if inst.mltype == "segmentation":
        return SegmentationArray
    return ObjDetectionArray

