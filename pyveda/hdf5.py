import os
import numpy as np
import tables
from pyveda.utils import mktempfilename

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

def _atom_from_dtype(_type):
    if isinstance(_type, np.dtype):
        return tables.Atom.from_dtype(_type)
    return tables.Atom.from_dtype(np.dtype(_type))


class LabelNotSupported(NotImplementedError):
    pass

class FrameworkNotSupported(NotImplementedError):
    pass


class WrappedDataArray(object):
    def __init__(self, node, trainer, output_transform=lambda x: x):
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


class ClassificationArray(WrappedDataArray):
    def _input_fn(self, item):
        dims = item.shape
        if len(dims) == 2:
            return item # for batch append stacked arrays
        return item.reshape(1, *dims)

    @classmethod
    def create_array(cls, trainer, group, dtype):
        trainer._fileh.create_earray(group, "labels",
                                     atom = tables.UInt8Atom(),
                                     shape = (0, len(trainer.klass_map)))



class SegmentationArray(ImageArray):
    _default_dtype = np.float32
    @classmethod
    def create_array(cls, trainer, group, dtype):
        if not dtype:
            dtype = cls._default_dtype
        trainer._fileh.create_earray(group, "labels",
                                     atom = _atom_from_dtype(dtype),
                                     shape = tuple([s if idx > 0 else 0 for idx, s in enumerate(trainer.image_shape)]))


class DetectionArray(WrappedDataArray):
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
        return self._trainer._label_array_factory(self._node.labels, self._trainer)

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
              "detection": DetectionArray}

data_groups = {"TRAIN": "Data designated for model training",
               "TEST": "Data designated for model testing",
               "VALIDATION": "Data designated for model validation"}


class ImageTrainer(object):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname=None, klass_map=None,  framework=None,
                 title="Unknown", image_shape=(3, 256, 256), image_dtype=np.float32,
                 label_dtype=None, mltype="classification", append=True):

        if fname is None:
            fname = mktempfilename(prefix="veda", suffix='h5')

        self._framework = framework
        self._fw_loader = lambda x: x
        self._image_klass = ImageArray
        self._label_klass = mltype_map[mltype]
        self.image_shape = image_shape
        self.klass_map = klass_map

        if os.path.exists(fname):
            if not append:
                os.remove(fname)
            else:
                self._fileh = tables.open_file(fname, mode="a")
                return

        self._fileh = tables.open_file(fname, mode="a", title=title)
        for name, desc in data_groups.items():
            self._fileh.create_group("/", name.lower(), desc)

        classifications = {klass: tables.UInt8Col(pos=idx + 2) for idx, (klass, _)
                            in enumerate(sorted(data_groups.items(), key=lambda x: x[1]))}
        classifications["image_chunk_index"] = tables.UInt8Col(shape=2, pos=1)

        groups = {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}
        for name, group in groups.items():
            self._fileh.create_table(group, "hit_table", classifications,
                                    "Chip Index + Klass Hit Record", tables.Filters(0))
            self._image_klass.create_array(self, group, image_dtype)
            self._label_klass.create_array(self, group, label_dtype)


    def _image_array_factory(self, *args, **kwargs):
        return self._image_klass(*args, **kwargs)

    def _label_array_factory(self, *args, **kwargs):
        return self._label_klass(*args, **kwargs)

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
    def validation(self):
        return WrappedDataNode(self._fileh.root.validation, self)

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


