from pyveda.vedaset.abstract import *
from pyveda.fetch.handlers import get_label_handler, bytes_to_array
from pyveda.exceptions import NotSupportedException

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

    def __init__(self, mltype, classes, image_shape, image_dtype,
                 count=None, partition=[70, 20, 10]):
        self._mltype = mltype
        self._classes = classes
        self._image_shape = image_shape
        self._image_dtype = image_dtype
        self._count = count
        self._partition = partition
        self._configure_instance()

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

    @property
    def partition(self):
        return self._partition

    @partition.setter
    def partition(self, prt):
        assert(isinstance(prt, list))
        assert(len(prt) == 3)
        assert(sum(prt) == 100)
        self._partition = prt

    def _configure_instance(self):
        self._train = None
        self._test = None
        self._valiate = None
        self._img_arr_ = None
        self._lbl_arr_ = None

    def _configure_fetcher(self):
        pass



