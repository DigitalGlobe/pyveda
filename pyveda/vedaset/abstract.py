from abc import ABC, abstractmethod
from pyveda.exceptions import MLTypeError


class MetaDataSchema(type):
    def __instancecheck__(cls, obj):
        for dsc in cls._descriptors:
            try:
                v = cls.__getattribute__(obj)
            except Exception:
                return False
        return True


class BaseDataSchema(metaclass=MetaBaseSchema): pass


class DataSampleSchema(BaseDataSchema):
    _descriptors = ["count",
                    "mltype",
                    "classes",
                    "partition",
                    "image_shape",
                    "image_dtype"]


class MetaMLtype(type):
    def __subclasscheck__(cls, C):
        if super().__subclasscheck__(C.__class__):
            return True
        if hasattr(C, "mltype"):
            if super().__subclasscheck__(C.mltype.__class__):
                return True
            if isinstance(C.mltype, str):
                return cls._match_from_string(C.mltype)
        return False

    __instancecheck__ = __subclasscheck__


class MLtype(metaclass=MetaMLtype):
    @classmethod
    def _match_from_string(cls, mlstr):
        if mlstr.lower() == cls.__name__.lower():
            return True
        if mlstr.lower() == getattr(cls, "name", "").lower():
            return True
        return False


class BinaryClassificationType(MLtype):
    name = "classification"


class InstanceSegmentationType(MLtype):
    name = "segmentation"


class ObjectDetectionType(MLtype):
    name = "obj_detection"


_metatype_map = dict([(meta.name, meta) for meta in
                      [BinaryClassificationType,
                       InstanceSegmentationType,
                       ObjectDetectionType]])

class mltypes:
    """
    Intended for usage as main mltype api framework
    """
    mltype = MLtype
    metatype_map = _metatype_map

    @classmethod
    def from_string(cls, s, *args, **kwargs):
        typname = s.lower()
        try:
            metatype = cls.metatype_map[typname]
        except KeyError:
            raise MLTypeError("Cast error: unrecognized mltype string name '{}'".format(typname))
        return metatype(*args, **kwargs)


