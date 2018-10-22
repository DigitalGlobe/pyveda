import os
from collections import OrderedDict

import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported


FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

class WrappedDataArray(object):
    def __init__(self, array, trainer, output_transform=lambda x: x):
        self._arr = node
        self._trainer = trainer
        self._read_transform = output_transform

    def _input_fn(self, item):
        return item

    def _output_fn(self, item):
        return item

    def __iter__(self, spec=slice(None)):
        if isinstance(spec, slice):
            for rec in self._arr.iterrows(spec.start, spec.stop, spec.step):
                yield self._read_transform(self._output_fn(rec))
        else:
            for rec in self._arr[spec]:
                yield self._read_transform(self._output_fn(rec))

    def __getitem__(self, spec):
        if isinstance(spec, slice):
            return list(self.__iter__(spec))
        elif isinstance(spec, int):
            return self._read_transform(self._output_fn(self._arr[spec]))
        else:
            return self._arr[spec] # let pytables throw the error

    def __setitem__(self, key, value):
        raise NotSupportedException("For your protection, overwriting raw data in ImageTrainer is not supported.")

    def __len__(self):
        return len(self._arr)

    def append(self, item):
        self._arr.append(self._input_fn(item))

    @classmethod
    def create_array(cls, *args, **kwargs):
        raise NotImplementedError


class ImageArray(WrappedDataArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 4:
            return item # for batch append stacked arrays
        elif len(dims) in (2, 3): # extend single image array along axis 0
            return item.reshape(1, *dims)
        return item # what could this thing be, let it fail

    @classmethod
    def create_array(cls, trainer, group, dtype):
        shape = list(trainer.image_shape)
        shape.insert(0,0)
        trainer._fileh.create_earray(group, "images",
                                     atom = _atom_from_dtype(dtype),
                                     shape = tuple(shape))


class LabelArray(WrappedDataArray):
    def __init__(self, hit_table, *args, **kwargs):
        self._table = hit_table
        super(WrappedDataArray, self).__init__(*args, **kwargs)

    def _add_records(self, labels):
        records = [tuple([self._hit_test(label[klass])]) for klass in label) for label in labels]
        self._table.append(records)
        self._table.flush()

    def _hit_test(self, record):
        if isinstance(record, int) and record in (0, 1):
            return record
        elif isinstance(record, list):
            return len(record)
        else:
            raise ValueError("say something")

    def append(self, labels):
        super(LabelArray, self).append(labels)
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
                                     shape = (0, len(trainer.klasses)))


class SegmentationArray(LabelArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        dims = item.shape
        assert len(dims) in (2, 3)
        if len(dims) == 2:
            return item.reshape(1, *dims)
        return item

    @classmethod
    def create_array(cls, trainer, group, dtype):
        if not dtype:
            dtype = cls._default_dtype
        trainer._fileh.create_earray(group, "labels",
                                     atom = _atom_from_dtype(dtype),
                                     shape = tuple([s if idx > 0 else 0 for idx, s in enumerate(trainer.image_shape)]))


class ObjDetectionArray(LabelArray):
    _default_dtype = np.float32

    def _input_fn(self, item):
        assert item.shape[1] == 4
        # Detection Bboxes are np arrays of shape (N, 4)
        return item.flatten()

    def _output_fn(self, item):
        op_shape = (int(len(item) / 4), 4)
        return item.reshape(op_shape)

    @classmethod
    def create_array(cls, trainer, group, dtype):
        if not dtype:
            dtype = cls._default_dtype
        trainer._fileh.create_vlarray(group, "labels",
                                      atom = _atom_from_dtype(dtype))


class WrappedDataNode(object):
    def __init__(self, node, trainer):
        self._node = node
        self._trainer = trainer

    @property
    def images(self):
        return self._trainer._image_array_factory(self._node.images, self._trainer, output_transform = self._trainer._fw_loader)

    @property
    def labels(self):
        return self._trainer._label_array_factory(self._node.hit_table, self._node.labels,  self._trainer)

    def __getitem__(self, spec):
        if isinstance(spec, int):
            return [self.images[spec], self.labels[spec]]
        else:
            return list(self.__iter__(spec))

    def __iter__(self, spec=None):
        if not spec:
            spec = slice(0, len(self)-1, 1)
        gimg = self.images.__iter__(spec)
        glbl = self.labels.__iter__(spec)
        while True:
            yield (gimg.__next__(), glbl.__next__())

    def __len__(self):
        return len(self._node.images)


mltype_map = {"classification": ClassificationArray,
              "segmentation": SegmentationArray,
              "object_detection": ObjDetectionArray}

data_groups = {"TRAIN": "Data designated for model training",
               "TEST": "Data designated for model testing",
               "VALIDATE": "Data designated for model validation"}


class VedaBase(object):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname, mltype, klasses, image_shape, image_dtype, framework=None,
                 title="Unknown", label_dtype=None, overwrite=False):

        self._framework = framework
        self._fw_loader = lambda x: x
        self.image_shape = [s for s in image_shape if s != 0]
        self.klasses = klasses

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing()
                return

        self._fileh = tables.open_file(fname, mode="a", title=title)
        for name, desc in data_groups.items():
            self._fileh.create_group("/", name.lower(), desc)

        classifications = dict([(klass, tables.UInt8Col(pos=idx + 1)) for idx, klass in enumerate(klasses)])

        self._create_tables(classifications, filters=tables.Filters(0))
        self._create_arrays(ImageArray, image_dtype)
        self._create_arrays(mltype_map[mltype])

    def _configure_new(self, *args, **kwargs):
        pass

    def _image_array_factory(self, *args, **kwargs):
        return self._image_klass(*args, **kwargs)

    def _label_array_factory(self, *args, **kwargs):
        return self._label_klass(*args, **kwargs)

    def _create_arrays(self, data_klass, data_dtype=None):
        for name, group in self._groups.items():
            data_klass.create_array(self, group, data_dtype)

    def _create_tables(self, classifications, filters=tables.Filters(0)):
        for name, group in self._groups.items():
            self._fileh.create_table(group, "hit_table", classifications,
                                     "Label Hit Record", filters)

    def _build_label_tables(self, rebuild=True):
        pass

    @property
    def _groups(self):
        return {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, fw):
        if fw and fw not in FRAMEWORKS:
            raise FrameworkNotSupported("Image adaptor not supported for {}".format(fw))
        self._framework = fw
        self._fw_loader = lambda x: x # TODO: Custom loaders here

    @property
    def train(self):
        return WrappedDataNode(self._fileh.root.train, self)

    @property
    def test(self):
        return WrappedDataNode(self._fileh.root.test, self)

    @property
    def validate(self):
        return WrappedDataNode(self._fileh.root.validate, self)

    def flush(self):
        self._fileh.flush()

    def close(self):
        self._fileh.close()

    def remove(self):
        raise NotImplementedError

    def __repr__(self):
        return self._fileh.__str__()

    def __del__(self):
        self.close()


