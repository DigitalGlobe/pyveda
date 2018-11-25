from abc import ABC, abstractmethod

class BaseVedaSequence(ABC):
    """Low level data access api to homogeneous sequences of data in PyVeda"""

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def __setitem__(self, idx, val):
        raise NotImplementedError


class BaseVedaGroup(ABC):
    """Pair-wise access patterns defined on a group of BaseVedaSequences"""

    def __len__(self):
        return len(self.images)

    def __iter__(self, spec=None):
        if not spec:
            spec = slice(0, len(self)-1, 1)
        gimg = self.images.__iter__(spec)
        glbl = self.images.__iter__(spec)
        while True:
            yield (gimg.__next__(), glbl.__next__())

    def __getitem__(self, idx):
        if isinstance(spec, int):
            return [self.images[spec], self.labels[spec]]
        return list(self.__iter__(spec))

    def __setitem__(self, idx, val):
        raise NotImplementedError

    @property
    @abstractmethod
    def images(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError


class BaseVedaSet(ABC):
    """Core representation of partitioned Machine-Learning datasets in PyVeda"""

    @property
    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def mltype(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def classes(self):
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


