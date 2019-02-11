import os
from functools import partial
import numpy as np
import tables
from pyveda.utils import ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.base import BaseVariableArray, BaseSampleArray, BaseDataSet
from pyveda.frameworks.batch_generator import VedaStoreGenerator
from pyveda.vedaset.store.arrays import get_array_handler, NDImageMixin
from pyveda.fetch.aiohttp.client import VedaBaseFetcher
#from pyveda.vv.labelizer import Labelizer


ignore_NaturalNameWarning = partial(ignore_warnings,
                                    _warning=tables.NaturalNameWarning)

class HDF5PartitionedArray(BaseVariableArray):
    """
    This wraps a pytables array with access determined
    by a contiguous index range given by two integers
    """
    def __init__(self, vset, arr):
        super(HDF5PartitionedArray, self).__init__(vset, arr)
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
            return self._output_fn(self._arr.__getitem__(spec))
        elif isinstance(key, slice):
            spec = self._translate_slice(key)
        return [self._output_fn(v) for v in self._arr.__getitem__(spec)]

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
        return self._output_fn(self._itr.__next__())

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

    def append(self, item):
        self._arr.append(self._input_fn(item))

    def append_batch(self, items):
        items = [self._input_fn(v) for v in items]
        self._arr.append(items)


class WrappedSampleNode(BaseSampleArray):

    def batch_generator(self, batch_size, shuffle=True, channels_last=False, rescale=False, flip_horizontal=False, flip_vertical=False, **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaBase partition.
        #Arguments
            batch_size: int. batch size
            shuffle: Boolean.
            channels_last: Boolean. To return image data as Height-Width-Depth, instead of the default Depth-Height-Width
            rescale: boolean. Rescale image values between 0 and 1.
            flip_horizontal: Boolean. Horizontally flip image and lables.
            flip_vertical: Boolean. Vertically flip image and lables
        """
        return VedaStoreGenerator(self, batch_size=batch_size, shuffle=shuffle,
                                channels_last=channels_last, rescale=rescale,
                                flip_horizontal=flip_horizontal, flip_vertical=flip_vertical, **kwargs)

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

    def clean(self, count=None):
        """
        Page through VedaStream data and flag bad data.
        Params:
            count: the number of tiles to clean
        """
        classes = self._vset.classes
        mltype = self._vset.mltype
        Labelizer(self, mltype, count, classes).clean()


class H5DataBase(BaseDataSet):
    """
    An interface for consuming and reading local data intended to be used with
    machine learning training
    """
    _fetch_class = VedaBaseFetcher
    _sample_class = WrappedSampleNode
    _variable_class = HDF5PartitionedArray

    def __init__(self, fname, title="SBWM", overwrite=False, mode="a", **kwargs):

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode)
                return

        super(H5DataBase, self).__init__(**kwargs)

        self._build_filetree()

    def _load_existing(self, fname, mode="a"):
        if mode == "w":
            raise ValueError("Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode)

    def _build_filetree(self):
        # Build group nodes
        for name in self._groups:
            self._fileh.create_group("/", name.lower(),
                                     "Records of ML experimentation phases")
        # Build table, array leaves
        self._create_tables()
        self._create_arrays()

    @property
    def _root(self):
        return self._fileh.root

    @ignore_NaturalNameWarning
    def _create_arrays(self):
        self._img_arr.create_array(self, self._fileh.root, self.image_dtype)
        self._lbl_arr.create_array(self, self._fileh.root)

    @ignore_NaturalNameWarning
    def _create_tables(self, filters=tables.Filters(0)):
        # Table col specs in the structure pytables wants
        idx_cols = {"ids": tables.StringCol(36)}
        feature_cols = dict([(klass, tables.UInt8Col(pos=idx + 1))
                             for idx, klass in enumerate(self.classes)])

        # Build main id index and feature tables on root
        self._fileh.create_table(self._root, "sample_index", idx_cols,
                                 "Constituent Datasample ID index", filters)
        self._fileh.create_table(self._root, "feature_table", feature_cols,
                                 "Datasample feature contexts", filters)

        # Build tables on groups that can be used for recording id logs during
        # model experimentation
        for name, group in self._hgroups.items():
            self._fileh.create_table(group, "sample_log", idx_cols,
                                     "Datasample ID log", filters)

    @property
    def _img_arr(self):
        if self._img_arr_ is None:
            self._img_arr_ = NDImageArray(self, self._fileh.root.images)
        return self._img_arr_

    @property
    def _lbl_arr(self):
        if self._lbl_arr_ is None:
            lbl_handler_class = get_array_handler(self)
            self._lbl_arr_ = lbl_handler_class(self, self._fileh.root.labels)
        return self._lbl_arr_

    @property
    def _hgroups(self):
        return {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}

    def flush(self):
        self._fileh.flush()

    def close(self):
        self._fileh.close()

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

    @classmethod
    def from_vtype(cls, fname, **vtype):
        return cls(fname, **vtype)

