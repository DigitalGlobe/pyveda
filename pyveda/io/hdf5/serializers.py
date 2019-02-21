import os
import numpy as np
import tables
import ujson as json
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.abstract import mtypes


def H5StoreIOAdapter(vstore):
    config = {}


    @staticmethod
    def _input_fn(item):
        dims = item.shape
        if len(dims) == 4:
            return item # for batch append stacked arrays
        if len(dims) in (2, 3): # extend single image array along axis 0
            return item.reshape(1, *dims)
        return item # what could this thing be, let it fail

    @staticmethod
    def create_array(vset, group):
        atom = _atom_from_dtype(vset.image_dtype)
        shape = list(vset.image_shape)
        shape.insert(0,0)
        shape = tuple(shape)
        vset._fileh.create_earray(group, "images", atom=atom, shape=shape)


class ClassificationMixin(IOLabelMixin):

    @staticmethod
    def _input_fn(item):
        dims = item.shape
        if len(dims) == 2:
            return item # for batch append stacked arrays
        return item.reshape(1, *dims)

    @staticmethod
    def create_array(vset, group):
        atom = tables.UInt8Atom()
        shape = (0, len(vset.classes))
        vset._fileh.create_earray(group, "labels", atom=atom, shape=shape)


class SegmentationMixin(IOLabelMixin, NDImageMixin):

    @staticmethod
    def create_array(vset, group, dtype=np.uint8):
        atom = _atom_from_dtype(dtype)
        shape = tuple([s if idx > 0 else 0 for
                       idx, s in enumerate(vset.image_shape)])
        vset._fileh.create_earray(group, "labels", atom=atom, shape=shape)


class ObjDetectionMixin(IOLabelMixin):
    _default_dtype = np.float32

    @staticmethod
    def _input_fn(item):
        assert isinstance(item, list)
        return np.fromstring(json.dumps(item), dtype=np.uint8)

    @staticmethod
    def _output_fn(item):
        return json.loads(item.tostring())

    @staticmethod
    def append_batch(vset, items):
        for item in items:
            vset.append(item)

    @staticmethod
    def create_array(vset, group, filters=tables.Filters(complevel=0)):
        atom = tables.UInt8Atom()
        vset._fileh.create_vlarray(group, "labels", atom=atom, filters=filters)


class
def get_array_handler(inst):
    if inst.mltype == "classification":
        return ClassificationMixin
    if inst.mltype == "segmentation":
        return SegmentationMixin
    return ObjDetectionMixin

