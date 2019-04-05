import numpy as np
from collections import defaultdict, OrderedDict
from pyveda.exceptions import NotSupportedException
from pyveda.vedaset.props import register_vprops, VDATAPROPS, DATAGROUPS
from pyveda.vedaset.interface import is_iterator, slices_from_partition, is_partitionable
from pyveda.vedaset.interface import BaseVariableArray, OpRegister, RegisterCatalog


class BaseSampleArray(object):
    _proxied = ("mltype",
                "classes",
                "image_shape",
                "image_dtype",)

    def __init__(self, vset, group):
        self._vset = vset
        self._group = group
        self.images = vset._img_factory(group)
        self.labels = vset._lbl_factory(group)

    def __getattr__(self, name):
        if name in self._proxied:
            return self._vset.__getattribute__(name)
        raise AttributeError

    @property
    def allocated(self):
        return self.images.allocated

    @property
    def images(self):
        if self._images is None:
            self._images = self._vset._img_factory(self._group)
        return self._images

    @images.setter
    def images(self, images):
        self._images = images

    @property
    def labels(self):
        if self._labels is None:
            self._labels = self._vset._lbl_factory(self._group)
        return self._labels

    @labels.setter
    def labels(self, labels):
        self._labels = labels

    def __getitem__(self, key):
        return [self.images[key],
                self.labels[key]]

    def __len__(self):
        return self.images.allocated

    def __iter__(self):
        return self

    def __next__(self):
        return [self.images.__next__(),
                self.labels.__next__()]



@register_vprops()
class BaseDataSet(object):
    _vprops = VDATAPROPS
    _sample_class = BaseSampleArray
    _variable_class = BaseVariableArray

    def __init__(self, groups=DATAGROUPS, **kwargs):
        self.groups = groups
        self._train = None
        self._test = None
        self._validate = None
        self._vidx_ = None

        self._prc = RegisterCatalog(OpRegister)
        self._register_prophooks()
        self._set_dprops(**kwargs)

        self._configure_instance()

    @classmethod
    def _set_dprop(cls, obj, dname, dval):
        try:
            p = getattr(cls, dname)
            p.__set__(obj, dval)
        except Exception as e:
            print(e)

    def _set_dprops(self, quiet=False, **kwargs):
        for k, v in kwargs.items():
            if k not in self._vprops.keys():
                if not quiet:
                    raise ValueError(
                        "Unexpected initialization argument!")
            self._set_dprop(self, k, v)

    def _configure_instance(self):
        if is_partitionable(self):
            self._update_indexes()

    def _register_prophooks(self):
        self._prc.count.register(self._update_indexes)
        self._prc.partition.register(self._update_indexes)

    @property
    def _vidx(self):
        if self._vidx_ is None:
            self._update_indexes()
        return self._vidx_

    def _update_indexes(self, *args):
        if not is_partitionable(self):
            self._vidx_ = None
            raise NotImplementedError
        self._vidx_ = OrderedDict()
        indexes = slices_from_partition(self.count, self.partition)
        for g, (start, stop) in zip(self.groups, indexes):
            self._vidx_[g] = (start, stop)
        self._train = None
        self._test = None
        self._validate = None

    def _unpack(self):
        """
        Instance method useful for unpacking an existing Vedatype-like thing
        into a map that can be used for initializing other Vedatype-like things
        """
        return dict([(pname, getattr(self, pname)) for pname
                     in self._vprops.keys()])

    def __len__(self):
        try:
            return len(self._image_array)
        except (TypeError, AttributeError):
            return self.count

    @property
    def train(self):
        if self._train is None:
            self._train = self._sample_class(self, "train")
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = self._sample_class(self, "test")
        return self._test

    @property
    def validate(self):
        if self._validate is None:
            self._validate = self._sample_class(self, "validate")
        return self._validate

    @property
    def _image_array(self):
        raise NotImplementedError

    @property
    def _label_array(self):
        raise NotImplementedError

    @property
    def _image_class(self):
        return self._variable_class

    @property
    def _label_class(self):
        return self._variable_class

    def _img_factory(self, group):
        arr = self._image_array
        return self._image_class(self, group, arr)

    def _lbl_factory(self, group):
        arr = self._label_array
        return self._label_class(self, group, arr)
