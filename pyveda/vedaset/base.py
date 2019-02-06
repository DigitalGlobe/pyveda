from collections import namedtuple
from pyveda.vedaset.abstract import *
from pyveda.fetch.handlers import get_label_handler, bytes_to_array
from pyveda.exceptions import NotSupportedException
from pyveda.vedaset.props import VSPropMap, is_iterator, DATAGROUPS


class VirtualIndexManager(object):
    IndexObj = namedtuple("IndexObj", "name start stop allocated")

    def __init__(self, partition, count, groups):
        self._partition = partition
        self._count = count
        self._groups = groups # This prescribes allocation ordering
        self.set_indexes()

    def __getitem__(self, key):
        if isinstance(key, int) and key in range(3):
            return self.indexes[self._groups[key]]
        if isinstance(key, str) and key in self._groups:
            return self.indexes[key]
        raise ValueError("Provide a data group name or order index")

    def set_indexes(self):
        n0, n1, n2 = [round(self.count * (p * 0.01)) for p in self.partition]
        idxs = [(0, n0, n0), (n0, n0 + n1, n1), (n0 + n1, self.count, n2)]
        # Set dict to proxy key val lookup access
        self.indexes = dict()
        for group, (start, stop, allocated) in zip(self._groups, idxs):
            self.indexes[group] = self.IndexObj(name=group, start=start,
                                           stop=stop, allocated=allocated)
        # Set instance vars for attribute access
        for g, idx in self.indexes.items():
            setattr(self, g, idx)

    @property
    def partition(self):
        return self._partition

    @partition.setter
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
        # TODO Check arr iterable/iterator container-like w append(write)
        self._arr_ = arr

    @property
    def _arr(self):
        return self._arr_

    @_arr.setter
    def _arr(self, arr):
        if not isinstance(arr, self.__class__)


class BaseSampleArray(ABCSampleIterator):
    _vtyp = "BaseSampleArray"

    def __init__(self, vset, group):
        self._vset = vset
        self._dgroup = group

    @property
    def mltype(self):
        return self._vset.mltype

    @property
    def classes(self):
        return self._vset.classes

    @property
    def image_shape(self):
        return self._vset.image_shape

    @property
    def image_dtype(self):
        return self._vset.image_dtype

    @property
    def allocated(self):
        return self._vset._vim[self._dgroup].allocated

    @property
    def images(self):
        return self._vset._img_arr

    @property
    def labels(self):
        return self._vset._lbl_arr

    @images.setter
    def images(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    @labels.setter
    def labels(self, val):
        raise NotSupportedException("Array modification denied: Protecting dataset integrity")

    def batch_iter(self, batch_size):
        while True:
            batch = []
            while len(batch) < batch_size:
                batch.append(self.__next__())
            yield batch

    def __getitem__(self, key):
        return [self.images[key], self.labels[key]]

    def __len__(self):
        return self.allocated

    def __iter__(self):
        raise NotImplementedError

    def __next__(self):
        raise NotImplementedError


class BaseDataSet(ABCDataSet):
    _vtyp = "BaseDataSet"
    _groups = DATAGROUPS
    _prop_map = VSPropMap
    _fetch_class = NotImplemented
    _sample_class = NotImplemented
    _variable_class = NotImplemented

    def __init__(self, **kwargs):
        # Set up vedasetprops from kwargs if given
        # Otherwise enforce typeset to limit available methods TODO
        # Define a call signature here using inspect TODO
        for pname, pdesc in self._prop_map.items():
            setattr(self, pname, pdesc.__init__())

        for pname, val in kwargs.items():
            if pname not in self._prop_map:
                raise ValueError("Unexpected initialization argument!")
            try:
                setattr(self, pname, val)
            except Exception as e:
                if val is not None:
                    raise

        self._configure_instance()

    def _configure_instance(self):
        self._train = None
        self._test = None
        self._valiate = None
        self._img_arr_ = None
        self._lbl_arr_ = None
        self.__prop_hooks__ = [] # Register prop hooks here
        try:
            self._vim = VirtualIndexManager(partition, count, groups)
        except AttributeError as ae:
            self._vim = None

    @property
    def _img_handler_class(self):
        return bytes_to_array

    @property
    def _lbl_handler_class(self):
        return get_label_handler(self)

    @property
    def train(self):
        if self._train is None:
            self._train = self._sample_class(self, "train")
        return self._train

    @property
    def test(self):
        if self._test is None:
            self._test = self._sample_class(self, "test")
        return self._test

    @property
    def validate(self):
        if self._validate is None:
            self._validate = self._sample_class(self, "validate")
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

    def _unpack(self):
        """
        Instance method useful for unpacking an existing Vedatype-like thing
        into a map that can be used for initializing other Vedatype-like things
        """
        return dict([(pname, getattr(self, pname)) for pname
                     in self._props.keys()])

