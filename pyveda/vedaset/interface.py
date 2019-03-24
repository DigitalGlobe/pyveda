from collections import OrderedDict
import numpy as np

# Modified from pandas
def is_iterator(obj):
    """
    Check if the object is an iterator.
    For example, lists are considered iterators
    but not strings or datetime objects.
    Parameters
    ----------
    obj : The object to check
    Returns
    -------
    is_iter : bool
        Whether `obj` is an iterator.
    Examples
    --------
    >>> is_iterator([1, 2, 3])
    True
    >>> is_iterator(datetime(2017, 1, 1))
    False
    >>> is_iterator("foo")
    False
    >>> is_iterator(1)
    False
    """

    if not hasattr(obj, '__iter__'):
        return False

    # Python 3 generators have
    # __next__ instead of next
    return hasattr(obj, '__next__')



def is_partitionable(obj):
    if not getattr(obj, "count", None):
        try:
            assert len(obj) > 0
        except Exception:
            return False
    if not getattr(obj, "partition", None):
        return False
    return True



def slices_from_partition(total, partition):
    # partition sums to 100
    allocations = [np.rint(total * p * 0.01) for p in partition]
    nparts = len(partition)
    idxs = []
    start = 0
    for i, alloc in enumerate(allocations):
        stop = start + alloc if i + 1 < nparts else total
        idxs.append((start, stop))
        start = stop
    return idxs


class RegisterCatalog(object):
    def __init__(self, factory):
        self.factory = factory

    def __getattr__(self, attr):
        self.__dict__[attr] = self.factory()
        return self.__dict__[attr]


class OpRegister(object):
    def __init__(self):
        self._ops_ = OrderedDict()

    def register(self, f, name=None, index=None):
        assert callable(f)
        if not name:
            name = getattr(f, "__name__", None) or f.__func__.__name__
        if index is None or index >= len(self._ops_):
            self._ops_[name] = f
            return
        ops = list(self._ops_.items())
        ops.insert(index, (name, f))
        self._ops_ = OrderedDict(ops)

    def clear(self):
        self._ops_ = OrderedDict()

    @property
    def _ops(self):
        return self._ops_.values()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._ops_.__getitem__(key)
        if isinstance(key, int):
            return list(self._ops_.values())[key]
        raise ValueError("Indexing supports str or int types only")

    def __iter__(self):
        return iter(self._ops)


class TransformRegister(OpRegister):
    def __init__(self, transforms=[], attach=False):
        super(TransformRegister, self).__init__()
        self._attached = attach
        for name, f in transforms:
            self.register(f, name)

    def attach(self):
        self._attached = True

    def detach(self):
        self._attached = False

    @property
    def _ops(self):
        if self._attached:
            return self._ops_.values()
        return []

    def transform(self, arg):
        _output = arg
        for fn in self:
            _output = fn(_output)
        return _output



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

    def append_batch(self, objs):
        for obj in objs:
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


class PartitionedIndexArray(BaseVariableArray):

    def __getitem__(self, key):
        if isinstance(key, int):
            spec = self._translate_idx(key)
        elif isinstance(key, slice):
            spec = self._translate_slice(key)
        else:
            raise NotImplementedError(
                "Numpy fancy indexing not supported")
        return super().__getitem__(spec)

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


class SerializedVariableArray(BaseVariableArray):
    def __init__(self, input_fn=lambda x: x,
                 output_fn=lambda x: x, *args, **kwargs):

        super(SerializedVariableArray, self).__init__(*args, **kwargs)
        self._ipf = input_fn
        self._opf = output_fn

    def _gettr(self, obj):
        obj = self._opf(obj)
        return super()._gettr(obj)

    def _settr(self, obj):
        obj = self._ipf(obj)
        return super()._settr(obj)


class ArrayTransformPlugin(BaseVariableArray):
    def __init__(self, *args, **kwargs):
        super(ArrayTransformPlugin, self).__init__(*args, **kwargs)
        self.transforms = TransformRegister()

    def _gettr(self, obj):
        obj = self.transforms.transform(obj)
        return super()._gettr(obj)


