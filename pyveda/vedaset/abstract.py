from abc import ABC, abstractmethod

class ABCDataVariable(ABC):

    _type = "ABCDataVariale"
    pass


class ABCDataSample(ABC):

    _typ = "ABCDataSample"
    pass


class ABCMetaCollection(ABC):

    _typ = "ABCMetaCollection"

    @property
    def mltype(self):
        raise NotImplementedError

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def imshape(self):
        raise NotImplementedError


class ABCVariableIterator(ABC):
    """Low level data access api to homogeneous sequences of data in PyVeda"""
    _typ = "ABCVariableIterator"

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError


class ABCSampleIterator(ABC):
    """Pair-wise access patterns defined on a group of BaseVedaSequences"""
    _typ = "ABCSampleIterator"

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


class ABCVariableArray(ABCVariableIterator):

    _typ = "ABCVariableArray"

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, item):
        raise NotImplementedError


class ABCSampleArray(ABCSampleIterator):

    _typ = "ABCSampleArray"

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError


class ABCDataSet(ABCMetaCollection):
    """Core representation of partitioned Machine-Learning datasets in PyVeda"""
    _typ = "ABCDataSet"

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


class ABCDataStore(ABCDataSet):

    _typ = "ABCDataStore"

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self):
        raise NotImplementedError


class BaseVedaCollection(ABCMetaCollection):

    _typ = "BaseVedaCollection"

    @property
    @abstractmethod
    def id(self):
        raise NotImplementedError

