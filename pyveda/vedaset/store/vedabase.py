import os
from functools import partial, wraps
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype, ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.store.arrays import ClassificationArray, SegmentationArray, ObjDetectionArray, NDImageArray
from pyveda.vedaset.abstract import BaseSampleArray, BaseDataSet
from pyveda.frameworks.batch_generator import VedaStoreGenerator


MLTYPE_MAP = {"classification": ClassificationArray,
              "segmentation": SegmentationArray,
              "object_detection": ObjDetectionArray}

DATA_GROUPS = {"TRAIN": "Data designated for model training",
               "TEST": "Data designated for model testing",
               "VALIDATE": "Data designated for model validation"}

ignore_NaturalNameWarning = partial(ignore_warnings, _warning=tables.NaturalNameWarning)


class WrappedDataNode(object):
    def __init__(self, node, trainer):
        self._node = node
        self._vset = trainer

    @property
    def images(self):
        return self._vset._image_klass(self._node.images, self._vset,
                                       output_transform=lambda x: x)

    @property
    def labels(self):
        return self._vset._label_klass(self._node.hit_table, self._node.labels,  self._vset)

    def batch_generator(self, batch_size, shuffle, **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaBase partition.
        #Arguments
            batch_size: int. batch size
            shuffle: boolean.
        """
        return VedaStoreGenerator(self, batch_size=batch_size, shuffle=shuffle, **kwargs)

    def __getitem__(self, spec):
        if isinstance(spec, int):
            return [self.images[spec], self.labels[spec]]
        else:
            return list(self.__iter__(spec))

    def __iter__(self, spec=None):
        if not spec:
            spec = slice(0, len(self), 1)
        gimg = self.images.__iter__(spec)
        glbl = self.labels.__iter__(spec)
        while True:
            yield (gimg.__next__(), glbl.__next__())

    def __len__(self):
        return len(self._node.images)



class H5DataBase(BaseDataSet):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname, mltype=None, klasses=None, image_shape=None,
                 image_dtype=None, title="SBWM", partition=[70, 20, 10],
                 overwrite=False, mode="a"):

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode, partition)
                return

        self._fileh = tables.open_file(fname, mode="a", title=title)
        self._fileh.root._v_attrs.mltype = mltype
        self._fileh.root._v_attrs.klasses = klasses
        self._fileh.root._v_attrs.image_shape = image_shape
        self._fileh.root._v_attrs.image_dtype = image_dtype
        self._fileh.root._v_attrs.partition = partition

        self._configure_instance()
        self._build_filetree()

    def _load_existing(self, fname, mode="a"):
        if mode == "w":
            raise ValueError("Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode)
        self._configure_instance()

    def _configure_instance(self, *args, **kwargs):
        self._image_klass = NDImageArray
        self._label_klass = MLTYPE_MAP[self.mltype]
        self._classifications = dict([(klass, tables.UInt8Col(pos=idx + 1)) for idx, klass in enumerate(self.classes)])

    def _build_filetree(self, dg=DATA_GROUPS):
        # Build group nodes
        for name, desc in dg.items():
            self._fileh.create_group("/", name.lower(), desc)
        # Build table, array leaves
        self._create_tables(self._classifications, filters=tables.Filters(0))
        self._create_arrays(self._image_klass, self.image_dtype)
        self._create_arrays(self._label_klass)

    @ignore_NaturalNameWarning
    def _create_arrays(self, data_klass, data_dtype=None):
        data_klass.create_array(self, self._fileh.root, data_dtype)

    @ignore_NaturalNameWarning
    def _create_tables(self, classifications, filters=tables.Filters(0)):
        # Build main id index and feature tables on root
        self._fileh.create_table(self._fileh.root, "sample_index",
                                 {"ids": tables.StringCol(36, pos=0)},
                                 "Constituent Datasample ID index", filters)
        self._fileh.create_table(self._fileh.root, "feature_table",
                                 classifications, "Datasample feature contexts",
                                 filters)

        # Build tables on groups that can be used for recording id logs during
        # model experimentation
        for name, group in self._groups.items():
            self._fileh.create_table(group, "sample_log",
                                     {"ids": tables.StringCol(36, pos=0)},
                                     "Datasample ID log", filters)

    @property
    def partition(self):
        return self._fileh.root._v_attrs.partition

    @property
    def mltype(self):
        return self._fileh.root._v_attrs.mltype

    @property
    def classes(self):
        return self._fileh.root._v_attrs.klasses

    @property
    def image_shape(self):
        return self._fileh.root._v_attrs.image_shape

    @property
    def image_dtype(self):
        return self._fileh.root._v_attrs.image_dtype

    @property
    def _groups(self):
        return {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}

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

    def __len__(self):
        return len(self._fileh.root.images)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return self._fileh.__str__()

    def __del__(self):
        self.close()

    @classmethod
    def from_path(cls, fname, **kwargs):
        inst = cls(fname, **kwargs)
        return inst

    #@classmethod
    #def from_vc(cls, vc, **kwargs):
    #    # Load an empty H5DataBase from a VC
