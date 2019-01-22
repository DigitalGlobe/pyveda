import os
from collections import OrderedDict, defaultdict
import numpy as np
import tables
import ujson as json
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.abstract import BaseVariableArray

class WrappedDataArray(BaseVariableArray):
    @staticmethod
    def _batch_transform(items):
        return np.array(items)

    def append_batch(self, items):
        self.append(items)

    def create_array(self, *args, **kwargs):
        raise NotImplementedError


class NDImageArray(WrappedDataArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 4:
            return item # for batch append stacked arrays
        elif len(dims) in (2, 3): # extend single image array along axis 0
            return item.reshape(1, *dims)
        return item # what could this thing be, let it fail

    def create_array(self, group, dtype):
        shape = list(trainer.image_shape)
        shape.insert(0,0)
        trainer._fileh.create_earray(group, "images",
                                     atom = _atom_from_dtype(dtype),
                                     shape = tuple(shape))


class LabelArray(WrappedDataArray):
    def __init__(self, hit_table, *args, **kwargs):
        self._table = hit_table
        super(LabelArray, self).__init__(*args, **kwargs)
        self.imshape = self._vset.image_shape

    def _add_records(self, labels):
        records = [tuple([self._hit_test(label[klass]) for klass in label]) for label in labels]
        self._table.append(records)
        self._table.flush()

    def _hit_test(self, record):
        if isinstance(record, int) and record in (0, 1):
            return record
        elif isinstance(record, list):
            return len(record)
        else:
            raise ValueError("say something")

    def append(self, label):
        super(LabelArray, self).append(label)
        #self._add_records(label)

    def append_batch(self, labels):
        return self.append(labels)
        #self._add_records(labels)

class ClassificationArray(LabelArray):
    _default_dtype = np.uint8

    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 2:
            return item # for batch append stacked arrays
        return item.reshape(1, *dims)

    @classmethod
    def create_array(cls, trainer, group, dtype):
        trainer._fileh.create_earray(group, "labels",
                                     atom = tables.UInt8Atom(),
                                     shape = (0, len(trainer.classes)))


class SegmentationArray(LabelArray):
    _default_dtype = np.float32

    @classmethod
    def create_array(cls, trainer, group, dtype):
        if not dtype:
            dtype = cls._default_dtype
        trainer._fileh.create_earray(group, "labels",
                                     atom = _atom_from_dtype(dtype),
                                     shape = tuple([s if idx > 0 else 0 for idx, s in enumerate(trainer.image_shape)]))


class ObjDetectionArray(LabelArray):
    _default_dtype = np.float32

    @staticmethod
    def _batch_transform(items):
        return items

    def _input_fn(self, item):
        assert isinstance(item, list)
        return np.fromstring(json.dumps(item), dtype=np.uint8)

    def _output_fn(self, item):
        return json.loads(item.tostring())

    def append_batch(self, items):
        for item in items:
            self.append(item)

    @classmethod
    def create_array(cls, trainer, group, dtype):
        if not dtype:
            dtype = cls._default_dtype
        trainer._fileh.create_vlarray(group, "labels",
                                      atom = tables.UInt8Atom(),
                                      filters=tables.Filters(complevel=0))



