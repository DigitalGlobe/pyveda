import os
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.db.arrays import ClassificationArray, SegmentationArray, ObjDetectionArray, ImageArray

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]


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
        self.image_shape = image_shape
        self.klasses = klasses
        self._image_klass = ImageArray
        self._label_klass = mltype_map[mltype]

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


