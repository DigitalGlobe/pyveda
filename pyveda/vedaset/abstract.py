from abc import ABC, abstractmethod


def create_veda_abc_type(name, attr):
    @classmethod
    def _check(cls, inst):
        return getattr(inst, attr)

    dct = dict(__instancecheck__=_check, __subclasshook__=_check)
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
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    #@abstractmethod
    #def __next__(self):
    #    raise NotImplementedError


class ABCSampleIterator(ABC):
    """Pair-wise access patterns defined on a group of BaseVedaSequences"""
    _vtyp = "ABCSampleIterator"

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self, spec):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def images(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError


class ABCDataSet(ABC):
    """Core representation of partitioned Machine-Learning datasets in PyVeda"""
    _vtyp = "ABCDataSet"

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def train(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def test(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def validate(self):
        raise NotImplementedError


class ABCMetaProps(ABC):

    _vtyp = "ABCMetaProps"

    @property
    @abstractmethod
    def mltype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self):
        raise NotImplementedError




