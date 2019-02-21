import numpy as np
from collections import defaultdict, OrderedDict
from pyveda.exceptions import NotSupportedException
from pyveda.vedaset.props import register_vprops, VBaseProps, DATAGROUPS
from pyveda.vedaset.interface import is_iterator, slice_from_partition,
                                        is_partitionable


class WrappedIterator(object):
    def __init__(self, arr):
        self._source = arr

    @property
    def _source(self):
        return self._source

    @_source.setter
    def _source(self, source):
        if not is_iterator(source):
            raise TypeError("Input source must be define iterator interface")
        self._source = source

    def __getitem__(self, obj):
        return self._source.__getitem__(obj)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._source)


class BaseVariableArray(WrappedIterator):
    def __init__(self, vset, group, arr):
        self._vset = vset
        self._group = group
        self._source = arr

    @property
    def _vidx(self):
        return getattr(self._vset._vidx, self._group)

    @property
    def _start(self):
        return self._vidx.start

    @property
    def _stop(self):
        return self._vidx.stop

    @property
    def allocated(self):
        return self._vidx.allocated

    def _gettr(self, obj):
        return obj

    def _settr(self, obj):
        return obj

    def append(self, obj):
        obj = self._settr(obj)
        self._source.append(obj)

    def __getitem__(self, key):
        obj = super(BaseVariableArray, self).__getitem__(key)
        if isinstance(key, int):
            return self._gettr(obj)
        return type(obj)([self._gettr(d) for d in obj])

    def __next__(self):
        obj = super(BaseVariableArray, self).__next__()
        return self._gettr(obj)

    def __len__(self):
        return self.allocated



class BaseSampleArray(object):
    _proxied = ("mltype",
                "classes",
                "image_shape",
                "image_dtype",)

    def __init__(self, vset, group):
        self._vset = vset
        self._group = group

    def __getattr__(self, name):
        if name in self._proxied:
            return self._vset.__getattribute__(name)
        raise AttributeError

    @property
    def images(self):
        return self._vset._image_array

    @property
    def labels(self):
        return self._vset._label_array

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
    _vpropmap = dict([(kls.__vname__, kls) for
                    kls in VBaseProps])
    _vprops = VBaseprops
    _sample_class = BaseSampleArray
    _variable_class = BaseVariableArray

    def __init__(self, groups=DATAGROUPS, **kwargs):
        self.groups = groups
        self._train = None
        self._test = None
        self._validate = None
        self._vidx_ = None

        self._set_dprops(**kwargs)
        self._prc = defaultdict(OpRegister)
        self._configure_instance()
        self._register_prophooks()

    @classmethod
    def _set_dprop(cls, obj, dname, dval):
        try:
            p = getattr(cls, dname)
            p.__set__(obj, dval)
        except Exception as e:
            pass

    def _set_dprops(self, quiet=False, **kwargs):
        for k, v in kwargs.items():
            if k not in self._vpropmap.keys():
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

    def _update_indexes(self):
        if not is_partitionable(self):
            self._vidx_ = None
            raise NotImplementedError
        self._vidx_ = OrderedDict()
        indexes = slices_from_partition(self.count, self.partition)
        for g, (start, stop) in zip(self.groups, indexes):
            self._vidx_[g] = PartitionIndex(start, stop)

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


