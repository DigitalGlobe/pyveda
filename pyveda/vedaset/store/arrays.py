import os
import numpy as np
import tables
import ujson as json
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported

class IOArrayMixin(object):
    pass


class IOImageMixin(IOArrayMixin):
    pass


class IOLabelMixin(IOArrayMixin):
    pass


class NDImageMixin(IOImageMixin):
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


class ClassificationMixin(IOLabelMixin):
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


class SegmentationMixin(IOLabelMixin):
    _default_dtype = np.float32

    def create_array(self, group):
        atom = _atom_from_dtype(self._vset.dtype)
        shape = tuple([s if idx > 0 else 0 for
                       idx, s in enumerate(self._vset.image_shape)])
        self._vset._fileh.create_earray(group, "labels", atom=atom, shape=shape)


class ObjDetectionMixin(IOLabelMixin):
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
        return ClassificationMixin
    if inst.mltype == "segmentation":
        return SegmentationMixin
    return ObjDetectionMixin

