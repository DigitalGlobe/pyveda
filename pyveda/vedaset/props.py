import inspect
from weakref import WeakKeyDictionary
from collections import defaultdict, OrderedDict
import numpy as np
import collections.abc
from pyveda.vedaset.abstract import mltypes

DATAGROUPS = ["train", "test", "validate"]

### class wrapper for descriptor assignment on class creation
def register_vprops(vprops=[], fallback="_vprops", exclude=[],):
    def wrapped(cls):
        for prop in set(vprops).union(set(getattr(cls, fallback, {}).values())):
            if prop.__vname__ not in exclude:
                setattr(cls, prop.__vname__, prop())
        return cls
    return wrapped

#### General data descriptor type system for all veda ml-accessors ####
class BaseDescriptor(object):
    __vname__ = NotImplemented

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if not getattr(self, "name", None):
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

    def __get__(self, instance, klass):
        val = super().__get__(instance, klass)
        try:
            return self.expected_type(val)
        except:
            return val

    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            try:
                value = self.expected_type(value)
            except Exception:
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
    registry_target = "_prc"

    def __set__(self, instance, value):
        super().__set__(instance, value)
        registry = getattr(instance, self.registry_target, None)
        if registry:
            for cb in registry[self.name]:
                if inspect.ismethod(cb):
                    cb(self.name, value)
                else:
                    cb(self.name, value)


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


class ImageShapedTyped(SizedMatched, PropCallbackExecutor):
    __vname__ = "image_shape"
    allowed_sizes = [2, 3]

    def __set__(self, instance, value):
        value = tuple(value)
        super().__set__(instance, value)


class MLtypeTyped(Typed, PropCallbackExecutor):
    __vname__ = "mltype"
    expected_type = mltypes.mltype

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = mltypes.from_string(value)
        super().__set__(instance, value)


class PartitionedTyped(SizedMatched, ProbabilityDistTyped, PropCallbackExecutor):
    __vname__ = "partition"
    allowed_sizes = [3]


class FeatureClassTyped(ListTyped, PropCallbackExecutor):
    __vname__ = "classes"


class SampleCountTyped(IntTyped, PropCallbackExecutor):
    __vname__ = "count"


_vdataprops = (SampleCountTyped,
              FeatureClassTyped,
              PartitionedTyped,
              MLtypeTyped,
              ImageShapedTyped,
              NumpyDataTyped,
              )

VDATAPROPS = dict([(kls.__vname__, kls) for kls in _vdataprops])
