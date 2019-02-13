from weakref import WeakKeyDictionary
from collections import defaultdict, OrderedDict
import numpy as np
import collections.abc
from pyveda.vedaset.abstract import mltypes

DATAGROUPS = ["train", "test", "validate"]

### class wrapper for descriptor assignment on class creation
def register_vprops(vprops=[], fallback="_vprops", exclude=[],):
    def wrapped(cls):
        for prop in props or getattr(cls, fallback, []):
            if prop.__vname__ not in exclude:
                setattr(cls, prop.__vname__, prop())
        return cls
    return wrapped

#### General data descriptor type system for all veda ml-accessors ####
class BaseDescriptor(object):
    __vname__ = NotImplemented

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not getattr(self, name, None):
            self.name = type(self).__vname__

    def __get__(self, instance, klass):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.name]
        except KeyError as ke:
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(type(instance).__name__, self.name)) from None

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


class Typed(BaseDescriptor):
    expected_type = type(None)

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError('Expected ' + str(self.expected_type))
        super().__set__(instance, value)


class BoolTyped(Typed):
    expected_type = bool


class IntTyped(Typed):
    expected_type = int


class StringTyped(Typed):
    expected_type = str


class SequenceTyped(Typed):
    expected_type = collections.abc.Sequence


class ListTyped(Typed):
    expected_type = list


class SizedMatched(SequenceTyped):
    allowed_sizes = []

    def __set__(self, instance, value):
        if len(value) not in self.allowed_sizes: # size desc? lol
            raise ValueError("Size '{}' must match '{}'".format(len(value), self.allowed_sizes))
        super().__set__(instance, value)


class ProbabilityDistTyped(SequenceTyped):
    def __set__(self, instance, value):
        if sum(value) != 100:
            raise ValueError("Probability distribution must sum to 100")
        super().__set__(instance, value)


class PropCallbackExecutor(BaseDescriptor):
    # This class should be mixed in last
    registry_target = "_prc"

    def __set__(self, instance, value):
        registry = getattr(instance, self.registry_target, None)
        if registery:
            for cb in registry[self.name].callbacks:
                cb(instance, value)
        super().__set__(instance, value)


class NumpyDataTyped(PropCallbackExecutor):
    __vname__ = "image_dtype"
    # Normal instance checking doesn't work here so Typed desc not usable
    def __set__(self, instance, value):
        try:
            d = np.dtype(value) # works on string identifiers and dtype instances
            assert d.name in np.sctypeDict
        except Exception as e:
            raise TypeError("Must provide np.dtype object or dtype castable str") from None
        super().__set__(instance, d)


class ImageShapedTyped(SizeMatched, PropCallbackExecutor):
    __vname__ = "image_shape"
    allowed_sizes = [2, 3]

    def __set__(self, instance, value):
        value = tuple(value)
        super().__setitem__(instance, value)


class MLtypeTyped(Typed, PropCallbackExecutor):
    __vname__ = "mltype"
    expected_type = mltypes.mltype

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = mltypes.from_string(value)
        super().__set__(instance, value)


class PartitionedTyped(SizeMatched, ProbabilityDistTyped, PropCallbackExecutor):
    __vname__ = "partition"
    size = [3]


class FeatureClassTyped(ListTyped, PropCallbackExecutor):
    __vname__ = "classes"


class SampleCountTyped(IntTyped, PropCallbackExecutor):
    __vname__ = "count"


#### Simple callback registry/catalog that can be utilized with CallbackExecutor
#### on any accessor classes to register arbitrary callbacks on any
#### descriptor.__set__ at global scale

class CallbackRegister(object):
    def __init__(self, d):
        self._d = d

    def register(self, fn, name=None):
        assert callable(fn)
        if name is None:
            name = fn.__name__
        self._d[name] = fn

    def unregister(self, iden):
        if callable(iden):
            iden = iden.__name__
        if isinstance(iden, str):
            try:
                self._d.__delitem__(iden)
            except KeyError:
                pass
            return
        raise TypeError("dunno that")

    @property
    def callbacks(self):
        return self._d.values()


class PropCallbackRegistery(object):
    def __init__(self, factory=OrderedDict, register=CallbackRegister, factory=OrderedDict):
        self._register = register
        self._cbindex = defaultdict(factory)

    def __getattr__(self, key):
        self.__dict__[key] = value = self._register(self._cbindex[key])
        return value


### Random utils, need a types api module maybe
# Modified rom pandas
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


