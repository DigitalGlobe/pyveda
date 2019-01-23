from collections import namedtuple
from pyveda.vedaset.abstract import *
from pyveda.fetch.handlers import get_label_handler, bytes_to_array
from pyveda.exceptions import NotSupportedException


class VirtualIndexManager(object):
    IndexObj = namedtuple("IndexObj", "name start stop allocated")
    order = ["train", "test", "validate"]

    def __init__(self, partition, count):
        self._partition = partition
        self._count = count
        self.set_indexes()

    def __getitem__(self, key):
        if isinstance(key, int) and key in range(3):
            return self.indexes[self.order[key]]
        if isinstance(key, str) and key in self.order:
            return self.indexes[key]
        raise ValueError("Provide a data group name or order index")

    def set_indexes(self):
        n0, n1, n2 = [round(self.count * (p * 0.01)) for p in self.partition]
        idxs = [(0, n0, n0), (n0, n0 + n1, n1), (n0 + n1, self.count, n2)]
        # Set dict to proxy key val lookup access
        self.indexes = dict()
        for group, (start, stop, allocated) in zip(self.order, idxs):
            self.indexes[group] = self.IndexObj(name=group, start=start,
                                           stop=stop, allocated=allocated)
        # Set instance vars for attribute access
        for g, idx in self.indexes.items():
            setattr(self, g, idx)

    @property
    def partition(self):
        return self._partition

    @parts.setter
    def partition(self, partition):
        self._partition = partition
        self.set_indexes()

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count):
        self._count = count
        self.set_indexes()


class BaseVariableArray(ABCVariableIterator):
    _vtyp = "BaseVariableArray"

    def __init__(self, vset, arr):
        self._vset = vset
        self._arr = iter(arr)

    def __getitem__(self, key):
        item = self._arr.__getitem__(key)
        return self._output_fn(item)

    def __iter__(self):
        return self._arr.__iter__()

    def __next__(self):
        item = self._arr.__next__()
        return self._output_fn(item)

    def __len__(self):
        return len(self._arr)

    def _input_fn(self, item):
        # Meant to be overriden when data must be transformed before writing,
        # for instance in the case of custom serialization
        return item

    def _output_fn(self, item):
        # Meant to be overridden when data must be transformed after reading,
        # for instance in the case of custom de-serialization
        return item

    def append(self, item):
        item = self._input_fn(item)
        return self._arr.append(item)



class BaseSampleArray(ABCSampleIterator):
    _vtyp = "BaseSampleArray"

    def __init__(self, vset):
        self._vset = vset

    @property
    def allocated(self):
        return self._vset._vim[self._dgroup].allocated

    @property
    def images(self):
        return self._vset._img_arr

    @images.setter
    def images(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    @property
    def labels(self):
        return self._vset._lbl_arr

    @labels.setter
    def labels(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    def __getitem__(self, key):
        return [self.images[key], self.labels[key]]

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError



class BaseDataSet(ABCDataSet, ABCMetaProps):
    _vtyp = "BaseDataSet"
    _fetch_class = NotImplemented
    _sample_class = NotImplemented
    _variable_class = NotImplemented

    def __init__(self, mltype, classes, image_shape, image_dtype,
                 count, partition=[70, 20, 10]):
        self._mltype = mltype
        self._classes = classes
        self._image_shape = image_shape
        self._image_dtype = image_dtype
        self._count = count
        self._partition = partition
        self._configure_instance()
        self._vim = VirtualIndexManager(partition, count)

    def _configure_instance(self):
        self._train = None
        self._test = None
        self._valiate = None
        self._img_arr_ = None
        self._lbl_arr_ = None

    @property
    def mltype(self):
        return self._mltype

    @property
    def classes(self):
        return self._classes

    @property
    def image_shape(self):
        return self._image_shape

    @property
    def image_dtype(self):
        return self._image_dtype

    @property
    def count(self):
        return self._count

    @count.setter
    def count(self, count):
        self._count = count
        self._vim.count = count

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, partition):
        assert(isinstance(partition, list))
        assert(len(partition) == 3)
        assert(sum(partition) == 100)
        self._partition = partition
        self._vim.partition = partition

    @property
    def _img_handler_class(self):
        return bytes_to_array

    @property
    def _lbl_handler_class(self):
        return get_label_handler(self)

    @property
    def train(self):
        if self._train is None:
            self._train = self._sample_class(self)
            setattr(self._train, "_dgroup", "train")
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = self._sample_class(self)
            setattr(self._test, "_dgroup", "test")
        return self._test

    @property
    def validate(self):
        if self._validate is None:
            self._validate = self._sample_class(self)
            setattr(self._validate, "_dgroup", "validate")
        return self._validate

    @property
    def _img_arr(self):
        raise NotImplementedError

    @property
    def _lbl_arr(self):
        raise NotImplementedError

    def _configure_fetcher(self):
        raise NotImplementedError

    def __len__(self):
        try:
            return len(self._img_arr)
        except (TypeError, AttributeError):
            return self.count


