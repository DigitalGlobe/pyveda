from abc import ABC, abstractmethod


def create_veda_abc_type(name, attr):
    @classmethod
    def _check(cls, inst):
        return getattr(inst, attr)

    dct = dict(__instancecheck__=_check, __subclasscheck__=_check)
    meta = type("ABCVtype", (type, ), dct)
    return meta(name, tuple(), dct)

ABCSourceArray = create_veda_abc_type("ABCSourceArray", "append")
# Fill out rest


class ABCDataVariable(ABC):

    _vtyp = "ABCDataVariale"
    pass


class ABCDataSample(ABC):

    _vtyp = "ABCDataSample"
    pass



class ABCVariableIterator(ABC):
    """Low level data access api to homogeneous sequences of data in PyVeda"""
    _vtyp = "ABCVariableIterator"

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
    _vtyp = "ABCSampleIterator"

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
    _vtyp = "ABCDataSet"

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


class ABCMetaProps(ABC):

    _vtyp = "ABCMetaProps"
    _props = ["mltype",
              "classes",
              "image_shape",
              "image_dtype",
              "count"]

    @property
    @abstractmethod
    def mltype(self):
        pass

    @property
    @abstractmethod
    def classes(self):
        pass

    @property
    @abstractmethod
    def image_shape(self):
        pass

    @property
    @abstractmethod
    def image_dtype(self):
        pass

    @property
    @abstractmethod
    def count(self):
        pass


class ABCMLType(type):
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


class MLType(metaclass=ABCMLType):
    @classmethod
    def _match_from_string(cls, mlstr):
        if mlstr.lower() == cls.__name__.lower():
            return True
        if mlstr.lower() == getattr(cls, "_mltype", "").lower():
            return True
        return False


class ClassificationType(MLType):
    _mltype = "Classification"


class SegmentationType(MLType):
    _mltype = "Segmentation"


class ObjectDetectionType(MLType):
    _mltype = "ObjDetection"


class mltypes:
    Classification = ClassificationType
    Segmentation = SegmentationType
    ObjectDetection = ObjectDetectionType



