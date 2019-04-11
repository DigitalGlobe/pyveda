import os
from functools import partial
import numpy as np
import tables
from pyveda.io.hdf5.serializers import adapt_serializers
from pyveda.vedaset.utils import ignore_NaturalNameWarning as ignore_nnw
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.base import BaseDataSet, BaseSampleArray
from pyveda.vedaset.interface import SerializedVariableArray, PartitionedIndexArray, ArrayTransformPlugin
from pyveda.frameworks.batch_generator import VedaStoreGenerator
#from pyveda.vv.labelizer import Labelizer


class H5VariableArray(SerializedVariableArray,
                      PartitionedIndexArray,
                      ArrayTransformPlugin):
    """
    This wraps a pytables array with access determined
    by a contiguous index range given by two integers
    """

    def __init__(self, vset, group, arr,
                 input_fn=None, output_fn=None):
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
        self._itr_ = self._arr.iterrows(self._start, self._stop)
        return self

    def __next__(self):
       try:
            return super().__next__()
       except StopIteration as si:
            self._itr_ = None
            raise



class H5SampleArray(BaseSampleArray):

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
            channels_last: Boolean. To return image data as Height-Width-Depth,
            instead of the default Depth-Height-Width
            rescale: boolean. Rescale image values between 0 and 1.
            flip_horizontal: Boolean. Horizontally flip image and lables.
            flip_vertical: Boolean. Vertically flip image and lables
        """
        return VedaStoreGenerator(self, batch_size=batch_size, shuffle=shuffle,
                                  channels_last=channels_last, rescale=rescale,
                                  flip_horizontal=flip_horizontal,
                                  flip_vertical=flip_vertical, **kwargs)

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
    _sample_class = H5SampleArray
    _variable_class = H5VariableArray

    def __init__(self, fname, title="SBWM", overwrite=False, mode="a", **kwargs):

        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode)
                return

        self._fileh = tables.open_file(fname, mode=mode, title=title)
        super(H5DataBase, self).__init__(**kwargs)
        self._build_filetree()

    def _configure_instance(self):
        self._image_class_ = None
        self._label_class_ = None
        self._image_array_ = None
        self._label_array_ = None
        super()._configure_instance()
        adapt_serializers(self)

    def _register_prophooks(self):
        super()._register_prophooks()
        wfn = lambda n, v: setattr(self._root._v_attrs, n, v)
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

    @ignore_nnw
    def _build_filetree(self):
        # Build group nodes
        for name in self.groups:
            self._fileh.create_group(
                "/", name.lower(), "Records of ML experimentation phases")

        self._image_class.create_array()
        self._label_class.create_array()
        self._build_tables()

    @classmethod
    def _build_tables(cls):
        pass

    @property
    def _attrs(self):
        return self._fileh.root._v_attrs

    @property
    def _root(self):
        return self._fileh.root

    @property
    def _image_class(self):
        if self._image_class_ is None:
            adapt_serializers(self)
        return self._image_class_

    @property
    def _label_class(self):
        if self._label_class_ is None:
            adapt_serializers(self)
        return self._label_class_

    @property
    def _image_array(self):
        return self._root.images

    @property
    def _label_array(self):
        return self._root.labels

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


class VedaBase(H5DataBase):

    class _MetaSample(tables.IsDescription):
        vid = tables.StringCol(36)

    @ignore_nnw
    def _build_tables(self):
        self._fileh.create_table(self._root,
                                 "metadata",
                                 self._MetaSample,
                                 "Veda Sample Metadata")

    @property
    def metadata(self):
        return self._root.metadata

    @classmethod
    def from_path(cls, fname, **kwargs):
        inst = cls(fname, **kwargs)
        return inst

    @classmethod
    def from_vtype(cls, fname, **vtype):
        return cls(fname, **vtype)


