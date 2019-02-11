from abc import ABC, abstractmethod
from pyveda.exceptions import MLTypeError


class ABCDataVariable(ABC):

    _vtyp = "ABCDataVariale"
    pass


class ABCDataSample(ABC):

    _vtyp = "ABCDataSample"
    pass



class ABCVariableIterator(ABC):
    """Low level data access api to homogeneous sequences of data in PyVeda"""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __iter__(self):
        pass


class ABCSampleIterator(ABC):
    """Pair-wise access patterns defined on a group of BaseVedaSequences"""

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, item):
        pass

    @abstractmethod
    def __iter__(self, spec):
        pass

    @abstractmethod
    def __next__(self):
        pass

    @property
    @abstractmethod
    def images(self):
        pass

    @property
    @abstractmethod
    def labels(self):
        pass


class ABCDataSet(ABC):
    """Core representation of partitioned Machine-Learning datasets in PyVeda"""

    @abstractmethod
    def __len__(self):
        pass

    @property
    @abstractmethod
    def train(self):
        pass

    @property
    @abstractmethod
    def test(self):
        pass

    @property
    @abstractmethod
    def validate(self):
        pass


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

    @classmethod
    @abstractmethod
    def _match_from_string(cls, strobj):
        pass

    __instancecheck__ = __subclasscheck__


class MLtype(metaclass=MetaMLtype):
    @classmethod
    def _match_from_string(cls, mlstr):
        if mlstr.lower() == cls.__name__.lower():
            return True
        if mlstr.lower() == getattr(cls, "_mltype", "").lower():
            return True
        return False


class ClassificationType(MLtype):
    _mltype = "classification"


class SegmentationType(MLtype):
    _mltype = "segmentation"


class ObjectDetectionType(MLtype):
    _mltype = "objdetection"


_metatype_map = dict([(meta._mltype, meta) for meta in
                      [ClassificationMLtype,
                       SegmentationMLtype,
                       ObjectDetectionMLtype]])

class mltypes:
    """
    Intended for usage as main mltype api framework
    """
    mltype = MLtype
    metatype_map = _metatype_map

    @classmethod
    def from_string(cls, s):
        typname = s.lower()
        try:
            metatype = cls.metatype_map[typname]
        except KeyError:
            raise MLTypeError("Cast error: unrecognized mltype string name '{}'".format(typname))
        return metatype()


