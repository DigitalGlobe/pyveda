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
from pyveda import vv
from pyveda.utils import update_options


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

    def __iter__(self):
        # Reset internal state
        self.images.__iter__()
        self.labels.__iter__()
        return self

    def batch_generator(self, batch_size, steps=None, loop=True, shuffle=True, channels_last=False, expand_dims=False, rescale=False,
                        flip_horizontal=False, flip_vertical=False, label_transform=None,
                        batch_label_transform=None, image_transform=None, pad=None, **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaBase partition.

        Args
            cache (VedaBase or VedaStream partition): Partition (train, test, or validate)
            batch_size (int): Number of samples in batch
            steps (int): Number of steps of batches to run in one epoch. If not provided, will calculate maximum possible number of complete batches
            loop (Boolean): Loop batcher indefinitely. If false, StopIteration is thrown after one epoch.
            shuffle (Boolean): Shuffle data between epochs.
            channels_last (Boolean): To return image data as Height-Width-Depth, instead of the default Depth-Height-Width
            rescale (Boolean): Return images rescaled to values between 0 and 1
            flip_horizontal (Boolean): Horizontally flip image and labels (50% probability)
            flip_vertical (Boolean): Vertically flip image and labels (50% probability)
            pad (int): Pad image with zeros to this dimension.
        """
        return VedaStoreGenerator(self, batch_size=batch_size, steps=steps, loop=loop, shuffle=shuffle,
                                channels_last=channels_last, expand_dims = expand_dims, rescale=rescale,
                                flip_horizontal=flip_horizontal, flip_vertical=flip_vertical,
                                label_transform=label_transform,
                                batch_label_transform=batch_label_transform,
                                image_transform=image_transform,
                                pad=pad, **kwargs)

    def clean(self, fname, count=None, include_background_tiles=True):
        """
        Page through VedaStream data and flag bad data.
        Params:
            count: the number of tiles to clean
        """
        classes = self._vset.classes
        mltype = self._vset.mltype
        vv.labelizer.Labelizer(self, mltype, count, classes, include_background_tiles, fname=fname).clean()

    def preview(self, count=10, include_background_tiles=True):
        classes = self._vset.classes
        mltype = self._vset.mltype
        vv.labelizer.Labelizer(self, mltype, count, classes, include_background_tiles).preview()

class H5DataBase(BaseDataSet):
    """
    An interface for consuming and reading local data intended to be used with
    machine learning training
    """
    _sample_class = H5SampleArray
    _variable_class = H5VariableArray
    _frozen = ("mltype", "image_shape", "image_dtype", "classes")

    def __init__(self, fname, title="SBWM", overwrite=False, mode="a", **kwargs):
        if not fname.endswith('.h5'):
            raise ValueError("filename must end in .h5")
        exists = False
        if os.path.exists(fname):
            if overwrite:
                os.remove(fname)
            else:
                exists = True
            if mode == "w" and exists:
                raise IOError(
                    "Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode, title=title)
        props = self._get_fprops()
        kwargs = update_options(props, kwargs, immutable=self._frozen)
        super(H5DataBase, self).__init__(**kwargs)
        if not exists:
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

    def _get_fprops(self):
        return dict([(name, self._attrs[name]) for name
                     in self._attrs._f_list()])

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

    def __init__(self, *args, dataset_id=None, **kwargs):
        super().__init__(*args, **kwargs)
        if dataset_id is not None:
            self._root._v_attrs["dataset_id"] = dataset_id

    def __iter__(self):
        img_iter = self._root.images.iterrows()
        lbl_iter = self._root.labels.iterrows()
        return(zip(img_iter, lbl_iter))

    def __next__(self):
       try:
            return super().__next__()
       except StopIteration as si:
            raise
            
    @ignore_nnw
    def _build_tables(self):
        self._fileh.create_table(self._root,
                                 "metadata",
                                 self._MetaSample,
                                 "Veda Sample Metadata")

    @property
    def metadata(self):
        return self._root.metadata

    @property
    def dataset_id(self):
        try:
            _id = self._root._v_attrs.dataset_id
        except AttributeError:
            return None
        return _id

    @classmethod
    def from_path(cls, fname, **kwargs):
        inst = cls(fname, **kwargs)
        return inst

    @classmethod
    def from_vtype(cls, fname, **vtype):
        return cls(fname, **vtype)
