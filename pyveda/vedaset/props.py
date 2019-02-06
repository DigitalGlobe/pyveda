from weakref import WeakKeyDictionary
import numpy as np
import collections.abc
from pyveda.vedaset.abstract import mltypes


DATAGROUPS = ["train", "test", "validate"]


class GenericNamedDescriptor(object):
    name = ""

    def __get__(self, inst, owner):
        if inst is None:
            return self
        if self.name not in inst.__dict__:
            raise AttributeError("Thing doesn't have {}".format(self.name))
        return inst.__dict__[self.name]

    def __set__(self, inst, val):
        inst.__dict__[self.name] = val


class CallbackHookDescriptor(GenericNamedDescriptor):
    def __set__(self, inst, val):
        for cb in getattr(inst, "__prop_hooks__", []):
            cb(val, self.name)
        super().__set__(inst, val)


class MLTypeDescriptor(CallbackHookDescriptor):
    name = "mltype"

    def __set__(self, inst, val):
        if isinstance(val, str):
            val = mltypes.from_string(val)
        if mltypes.is_mltype(val):
            super().__setitem__(inst, val)
        raise AttributeError("MLType not valid or cannot be inferred")


class ImageShapeDescriptor(CallbackHookDescriptor):
    name = "image_shape"

    def __get__(self, inst, owner):
        if inst is None:
            return self
        if self.name not in inst.__dict__:
            try:
                out = self._infer_shape(inst)
                self.__set__(inst, out)
                return out
            except Exception as e:
                raise AttributeError("Image shape not valid or cannot be inferred") from None
        return inst.__dict__[self.name]

    def __set__(self, inst, val):
        if not isinstance(val, collections.abc.Sequence):
            raise TypeError("Has to be sequence-like")
        if len(val) not in [2, 3]:
            raise TypeError("Unsupported Image dimension size, rank must be 2 or 3")
        val = tuple(val)
        super().__setitem__(inst, val)

    def _infer_shape(self, inst):
        raise NotImplementedError


class ImageDtypeDescriptor(CallbackHookDescriptor):
    name = "image_dtype"

    def __get__(self, inst, owner):
        if inst is None:
            return self
        if self.name not in inst.__dict__:
            try:
                out = self._infer_dtype(inst)
                self.__set__(inst, out)
                return out
            except Exception as e:
                raise AttributeError("Image dtype not valid or cannot be inferred") from None
        return inst.__dict__[self.name]

    def __set__(self, inst, val):
        if isinstance(val, str):
            val = np.dtype(val)
        # Better checks here
        super().__setitem__(inst, val)

    def _infer_dtype(self, inst):
        raise NotImplementedError


class MLClassDescriptor(CallbackHookDescriptor):
    name = "classes"

    def __get__(self, inst, owner):
        if inst is None:
            return self
        if self.name not in inst.__dict__:
            raise AttributeError("Label feature classes not defined")
        return inst.__dict__[self.name]

    def __set__(self, inst, val):
        if not isinstance(val, collections.abc.Sequence):
            raise TypeError("Gotta be list-like")
        if len(val) < 1:
            raise TypeError("Gotta have at least one thing in there")
        # Check that items are strings
        if not all([isinstance(f, str) for f in val]):
            raise TypeError("The things inside have to be strings come on")

        super().__setitem__(inst, val)


class DelegatedCountDescriptor(GenericNamedDescriptor):
    name = "count"

    # Delegates to VIM
    def __get__(self, inst, owner):
        if inst is None:
            return self
        return inst._vim.count

    def __set__(self, inst, val):
        assert isinstance(val, int)
        assert val > 0
        for cb in getattr(inst, "__prop_hooks__", []):
            cb(val, self.name)
        inst._vim.count = val


class DelegatedPartitionDescriptor(GenericNamedDescriptor):
    name = "partition"

    # Delegates to VIM
    def __get__(self, inst, owner):
        if inst is None:
            return self
        return inst._vim.partition

    def __set__(self, inst, val):
        assert(isinstance(val, list))
        assert(len(val) == 3)
        assert(sum(val) == 100)
        for cb in getattr(inst, "__prop_hooks__", []):
            cb(val, self.name)
        inst._vim.partition = partition


_vds = [MLTypeDescriptor,
        MLClassDescriptor,
        ImageShapeDescriptor,
        ImageDtypeDescriptor,
        DelegatedCountDescriptor,
        DelegatedPartitionDescriptor]

VSPropMap = {klass.name: klass for klass in _vds}


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


