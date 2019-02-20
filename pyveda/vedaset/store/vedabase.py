import os
from functools import partial
import numpy as np
import tables
from pyveda.vedaset.utils import ignore_NaturalNameWarning as ignore_nnw
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.base import BaseDataSet, BaseSampleArray, SerializedVariableArray,
                                PartitionedIndexArray, ArrayTransformPlugin
from pyveda.frameworks.batch_generator import VedaStoreGenerator
from pyveda.vedaset.store.arrays import get_array_handler, NDImageMixin
#from pyveda.vv.labelizer import Labelizer


class H5StoreVariableArray(SerializedVariableArray,
                           PartitionedIndexArray,
                           ArrayTransformPlugin):
    """
    This wraps a pytables array with access determined
    by a contiguous index range given by two integers
    """
    def __init__(self, vset, group, arr, input_fn=None, output_fn=None):
        super().__init__(vset, group, arr,
                         input_fn=input_fn,
                         output_fn=output_fn)

        self._itr_ = None

    @property
    def _itr(self):
        if self._itr_ is None:
            self._itr_ = self._arr.iterrows(self._start, self._stop)
        return self._itr_

    def __iter__(self):
        self._itr = self._arr.iterrows(self._start, self._stop)
        return self

    def __next__(self):
       try:
            return super().__next__()
        except StopIteration as si:
            self._itr_ = None
            raise


class H5StoreSampleArray(BaseSampleArray):

    def batch_generator(self, batch_size,
                        shuffle=True,
                        channels_last=False,
                        rescale=False,
                        flip_horizontal=False,
                        flip_vertical=False,
                        **kwargs):
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

    def __iter__(self):
        # Reset internal state
        self.images.__iter__()
        self.labels.__iter__()
        return self

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
    _sample_class = H5StoreSampleArray
    _variable_class = H5StoreVariableArray

    def __init__(self, fname, title="SBWM", overwrite=False, mode="a", **kwargs):

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode)
                return

        super(H5DataBase, self).__init__(**kwargs)
        self._fileh = tables.open_file(fname, mode=mode, title=title)

        self._build_filetree()

    def _configure_instance(self):
        super()._configure_instance()
        H5IOAdapter.configure(self)

    def _register_prophooks(self):
        super()._register_prophooks()
        wfn = lambda v, n: setattr(self._root._v_attrs, n, v)
        self._prc.mltype.register(wfn)
        self._prc.classes.register(wfn)
        self._prc.image_shape.register(wfn)
        self._prc.image_dtype.register(wfn)
        self._prc.partition.register(wfn)
        self._prc.count.register(wfn)

    def _load_existing(self, fname, mode="a"):
        if mode == "w":
            raise ValueError("Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode)
        p = dict([(name, self._attrs[name]) for name in self._attrs.f_list()])
        self._set_dprops(quiet=True, **p)
        self._register_prophooks()

    def _build_filetree(self):
        # Build group nodes
        for name in self.groups:
            self._fileh.create_group("/", name.lower(),
                                     "Records of ML experimentation phases")
        # Build table, array leaves
        self._create_tables()
        self._create_arrays()

    @property
    def _attrs(self):
        return self._fileh.root._v_attrs

    @property
    def _root(self):
        return self._fileh.root

    @ignore_nnw
    def _create_arrays(self):
        self._img_arr.create_array(self, self._fileh.root, self.image_dtype)
        self._lbl_arr.create_array(self, self._fileh.root)

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

