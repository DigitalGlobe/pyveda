import os
from functools import partial, wraps
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype, ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.base import BaseVariableArray, BaseSampleArray, BaseDataSet
from pyveda.frameworks.batch_generator import VedaStoreGenerator
from pyveda.vedaset.store.arrays import get_array_handler


ignore_NaturalNameWarning = partial(ignore_warnings, _warning=tables.NaturalNameWarning)


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

    def __iter__(self):
        # Reset internal state
        self.images.__iter__()
        self.labels.__iter__()
        return self

    def __next__(self):
        return [self.images.__next__(), self.labels.__next__()]


class H5DataBase(BaseDataSet):
    """
    An interface for consuming and reading local data intended to be used with
    machine learning training
    """
    def __init__(self, fname, image_dtype=None, title="SBWM", overwrite=False,
                 mode="a", *args, **kwargs):

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode, partition)
                return

        self._fileh = tables.open_file(fname, mode="a", title=title)
        self._fileh.root._v_attrs.mltype = mltype
        self._fileh.root._v_attrs.klasses = classes
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
        self._fileh.root._v_attrs.partition = prt
        self._partition = prt
        self._update_vindex()

    @property
    def _lbl_arr_class(self):
        return get_array_handler(self)

    @property
    def _img_arr_class(self):
        return NDImageArray

    @property
    def _img_arr(self):
        if self._img_arr_ is None:
            self._img_arr_ = self._img_arr_class(self, self._fileh.root.images)
        return self._img_arr_

    @property
    def _lbl_arr(self):
        if self._lbl_arr_ is None:
            self._lbl_arr_ = self._lbl_arr_class(self, self._fileh.root.labels)
        return self._lbl_arr_

    @property
    def _groups(self):
        return {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}

    @property
    def train(self):
        if self._train is None:
            self._train = WrappedSampleNode(self)
            setattr(self._train, "_dgroup", "train")
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = WrappedSampleNode(self)
            setattr(self._test, "_dgroup", "test")
        return self._test

    @property
    def validate(self):
        if self._validate is None:
            self._validate = WrappedSampleNode(self)
            setattr(self._validate, "_dgroup", "validate")
        return self._validate

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
