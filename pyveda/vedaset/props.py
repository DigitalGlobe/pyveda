from weakref import WeakKeyDictionary
from collections import defaultdict, OrderedDict
import numpy as np
import collections.abc
from pyveda.vedaset.abstract import mltypes

DATAGROUPS = ["train", "test", "validate"]

### class wrapper for descriptor assignment on class creation
def register_props(prop_map={}, exclude=[]):
    def wrapped(cls):
        props = prop_map.items() or getattr(cls, "_prop_map", {}).items():
            for name, prop in props:
                if name not in exclude:
                    setattr(cls, name, prop(name))
        return cls
    return wrapped

#### General data descriptor type system for all veda ml-accessors ####
###
class BaseDescriptor(object):
    name = NotImplemented

    def __init__(self, **opts):
        self.__dict__.update(opts)

    def __get__(self, instance, owner):
        if instance is None:
            return self
        try:
            return instance.__dict__[self.name]
        except KeyError as ke:
            raise AttributeError("'{}' object has no attribute '{}'"
                                 .format(type(self).__name__, self.name))

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value


# Descriptor for enforcing types
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
    def __init__(self, **opts):
        if not hasattr(self, size):
            if "size" not in opts:
                raise TypeError("Required input size option")
            self.size = opts['size']
        assert isinstance(self.size, (collections.abc.Sequence, int))
        super().__init__(**opts)

    def __set__(self, instance, value):
        if isinstance(self.size):
            if len(value) != self.size:
                raise ValueError("Size '{}' must equal '{}'".format(len(value), self.size))
        else:
            if len(value) not in self.size: # size desc? lol
                raise ValueError("Size '{}' must match '{}'".format(len(value), self.size))
        super().__set__(instance, value)


class UnityCheckedSum(SequenceTyped):
    def __set__(self, instance, value):
        if sum(value) != 100:
            raise ValueError("Probability distribution must sum to 100")
        super().__set__(instance, value)


class CallbackExecutor(BaseDescriptor):
    # This class should be mixed in last
    def __init__(self, register=None, **opts):
        if not register:
            register = getattr(self, "register", None) # try class attr
        self.register = register
        super().__init__(**opts)

    def __set__(self, instance, value):
        if self.register:
            registry = getattr(instance, self.register, None)
            if registery:
                callbacks = registery[self.name].callbacks
                for cb in callbacks:
                    cb(instance, value)
        super().__set__(instance, value)


class NumpyDataTyped(CallbackExecutor):
    # Normal instance checking doesn't work here so Typed desc not usable
    def __set__(self, instance, value):
        # This is the best way I could figure out how to check/cast np.dtypes
        try:
            d = np.dtype(value) # works on string identifiers and dtype instances
            if d.name not in np.sctypeDict:
                raise TypeError("Provided dtype must be one of np.sctypes")
        except TypeError as te:
            raise te
        except Exception as e:
            raise TypeError("Must provide np.dtype object or dtype castable str")  from e
        super().__set__(instance, d)


class NDArrayShaped(SizeMatched, CallbackExecutor):
    size = [2, 3]
    def __set__(self, instance, value):
        value = tuple(value)
        super().__setitem__(instance, value)


class MLTyped(Typed, CallbackExecutor):
    expected_type = mltypes.mltype

    def __set__(self, instance, value):
        if isinstance(value, str):
            value = mltypes.from_string(value)
        super().__set__(instance, value)


class DataPartitionTyped(SizeMatched, UnityCheckedSum, CallbackExecutor):
    size = 3


class FeatureClassTyped(ListTyped, CallbackExecutor):
    pass


class DataCountTyped(IntTyped, CallbackExecutor):
    pass

#### End data descriptor type definitions ####

#### Data type map configs for accessors ####
BaseDataTypeMap = {
                    "mltype": MLTyped,
                    "image_shape": NDArrayShaped,
                    "image_dtype": NumpyDataTyped,
                    "partition": DataPartitionTyped,
                    "count": DataCountTyped,
                    "classes": FeatureClassTyped
                  }


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


class RegisteryCatalog(object):
    def __init__(self, factory=OrderedDict, register=CallbackRegister):
        self._cat = defaultdict(factory)
        self._reg = register

    def __getattr__(self, key):
        self.__dict__[key] = value = self._reg(self._cat[key])
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


