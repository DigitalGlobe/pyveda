from pyveda.vedaset.abstract import ABCDataSet, ABCMetaProps
from pyveda.fetch.handlers import get_label_handler, bytes_to_array
from pyveda.exceptions import NotSupportedException

class BaseVariableArray(ABCVariableIterator):
    _vtyp = "BaseVariableArray"

    def __init__(self, vset, arr, data_transform=lambda x: x):
        self._vset = vset
        self._arr = arr
        self._data_transform = data_transform

    def __getitem__(self, key):
        item = self._arr.__getitem__(key)
        item = self._output_fn(item)
        return self.data_transform(item)

    def __iter__(self):
        return self._arr.__iter__()

    def __next__(self):
        item = self._arr.__next__()
        item = self._output_fn(item)
        return self.data_transform(item)

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

    @property
    def data_transform(self):
        return self._data_transform

    @data_transform.setter
    def data_transform(self, fn):
        if not callable(fn):
            raise ValueError("Transforms must be callable on data")
        self._data_transform = fn



class BaseSampleArray(ABCSampleIterator):
    _vtyp = "BaseSampleArray"

    def __init__(self, vset):
        self._vset = vset

    @property
    def images(self):
        return self._vset._img_arr

    @images.setter(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    @property
    def labels(self):
        return self._vset._lbl_arr

    @labels.setter(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    def __getitem__(self, key):
        return [self.images[key], self.labels[key]]

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError



class BaseDataSet(ABCDataSet, ABCMetaProps):
    _vtyp = "BaseDataSet"

    def __init__(self, mltype, classes, image_shape, count=None,
                 partition=[70, 20, 10]):
        self._mltype = mltype
        self._classes = classes
        self._image_shape = image_shape
        self._count = count
        self._partition = partition
        self._configure_instance()

    def _configure_instance(self):
        self._train = None
        self._test = None
        self._valiate = None

    def _configure_fetcher(self):
        pass


