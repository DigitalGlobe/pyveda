import os
from functools import partial, wraps
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype, ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.store.arrays import ClassificationArray, SegmentationArray, ObjDetectionArray, NDImageArray
from pyveda.vedaset.abstract import BaseVariableArray, BaseSampleArray, BaseDataSet
from pyveda.frameworks.batch_generator import VedaStoreGenerator


MLTYPE_MAP = {"classification": ClassificationArray,
              "segmentation": SegmentationArray,
              "object_detection": ObjDetectionArray}

ignore_NaturalNameWarning = partial(ignore_warnings, _warning=tables.NaturalNameWarning)

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
            self._start_ = self._vset._update_vindex(self)
        return self._start_

    @property
    def _stop(self):
        if self._stop_ is None:
            self._stop_ = self._vset._update_vindex(self)
        return self._stop_

    def __getitem__(self, key):
        if isinstance(key, int):
            spec = self._translate_idx(key)
        if isinstance(key, slice):
            spec = self._translate_slice(key)
        return self._arr.__getitem__(spec)

    def __iter__(self):
        return self

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


class WrappedSampleNode(BaseSampleArray):
    def batch_generator(self, batch_size, shuffle, **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaBase partition.
        #Arguments
            batch_size: int. batch size
            shuffle: boolean.  """
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

        self._build_filetree()

    def _load_existing(self, fname, mode="a"):
        if mode == "w":
            raise ValueError("Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode)

    def _build_filetree(self, dg=["train", "test", "validate"]):
        # Build group nodes
        for name in dg:
            self._fileh.create_group("/", name.lower(),
                                     "Records of ML experimentation phases")
        # Build table, array leaves
        self._create_tables(filters=tables.Filters(0))
        self._create_arrays(self._image_klass, self.image_dtype)
        self._create_arrays(self._label_klass)

    @ignore_NaturalNameWarning
    def _create_arrays(self, data_klass, data_dtype=None):
        data_klass.create_array(self, self._fileh.root, data_dtype)

    @ignore_NaturalNameWarning
    def _create_tables(self, filters=tables.Filters(0)):
        # Table col specs in the structure pytables wants
        idx_cols = {"ids": tables.StringCol(36)}
        feature_cols = dict([(klass, tables.UInt8Col(pos=idx + 1))
                             for idx, klass in enumerate(self.classes)])

        # Build main id index and feature tables on root
        self._fileh.create_table(self._fileh.root, "sample_index", idx_cols,
                                 "Constituent Datasample ID index", filters)
        self._fileh.create_table(self._fileh.root, "feature_table" feature_cols,
                                 "Datasample feature contexts", filters)

        # Build tables on groups that can be used for recording id logs during
        # model experimentation
        for name, group in self._groups.items():
            self._fileh.create_table(group, "sample_log", idx_cols,
                                     "Datasample ID log", filters)

    @property
    def partition(self):
        return self._fileh.root._v_attrs.partition

    @partition.setter
    def partition(self, prt):
        assert(isinstance(prt, list))
        assert(len(prt)) == 3
        assert(sum(prt) == 100)

        self._fileh.root._v_attrs.partition = prt
        self._update_vindex()

    @property
    def _img_arr(self):
        return self._fileh.root.images

    @property
    def _lbl_arr(self):
        return self._fileh.root.labels

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
        return WrappedSampleNode(self)

    @property
    def test(self):
        return WrappedSampleNode(self._fileh.root.test, self)

    @property
    def validate(self):
        return WrappedSampleNode(self._fileh.root.validate, self)

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
