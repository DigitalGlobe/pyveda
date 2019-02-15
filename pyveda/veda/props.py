import numpy as np


def prop_wrap(cls, prop_map, exclude=[]):
    for k, prop in prop_map.items():
        if k not in exclude:
            setattr(cls, k, prop(k))
    return cls


class GenericMappedProp(object):
    def __init__(self, name, val=None):
        self.val = val
        self.name = name

    def __get__(self, obj, objtype):
        if obj is None:
            return self
        return obj._meta[self.name]

    def __set__(self, obj, val):
        obj._meta[self.name] = val


class ListMappedProp(GenericMappedProp):
    def __init__(self, name, val=[]):
        super(ListMappedProp, self).__init__(name, val)

    def __set__(self, obj, val):
        if val not in obj._meta[self.name]:
            obj._meta.append(val)


class DtypeMappedProp(GenericMappedProp):
    def __get__(self, obj, objtype):
        if obj is None:
            return self
        dtype = obj._meta[self.name]
        if isinstance(dtype, str):
            dtype = np.dtype(dtype)
        return dtype

    def __set__(self, obj, val):
        if not isinstance(val, str):  # What's the best way to check for dtype again?
            val = val.__name__
        obj._meta[self.name] = val


class StrMappedProp(GenericMappedProp):
    pass


class IntMappedProp(GenericMappedProp):
    pass


class BoolMappedProp(GenericMappedProp):
    pass


VEDAPROPS = {"name": StrMappedProp,
             "classes": ListMappedProp,
             "dtype": DtypeMappedProp,
             "userId": StrMappedProp,
             "imshape": ListMappedProp,
             "releases": StrMappedProp,
             "mltype": StrMappedProp,
             "tilesize": ListMappedProp,
             "image_refs": ListMappedProp,
             "sensors": ListMappedProp,
             "bounds": ListMappedProp,
             "userId": StrMappedProp,
             "public": BoolMappedProp,
             "count": IntMappedProp,
             "percent_cached": IntMappedProp
             }
