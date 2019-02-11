from collections import namedtuple, defaultdict, OrderedDict
from pyveda.vedaset.abstract import *
from pyveda.fetch.handlers import get_label_handler, bytes_to_array
from pyveda.exceptions import NotSupportedException
from pyveda.vedaset.props import VSPropMap, is_iterator, DATAGROUPS
import numpy as np


def slices_from_partition(total, partition):
    # partition sums to 100
    allocations = [np.rint(total * p * 0.01) for p in partition]
    nparts = len(partition)
    idxs = []
    start = 0
    for i, alloc in enumerate(allocations):
        stop = start + alloc if i + 1 < nparts else total
        idxs.append((start, stop))
        start = stop
    return idxs


class VirtualIndexManager(object):
    IndexObj = namedtuple("IndexObj", "name start stop allocated")

    def __init__(self, partition, count, groups):
        self.partition = partition
        self.count = count
        self.groups = groups # This prescribes allocation ordering
        self.set_indexes()

    def __getitem__(self, key):
        if isinstance(key, int):
            try:
                return self.indexes[self.groups[key]]
            except IndexError as e:
                pass
        if isinstance(key, str):
            try:
                return self.indexes[key]
            except KeyError as e:
                pass
        else:
            raise TypeError("Get keys are String or integer")
        raise ValueError("Group name or index non existent") from e

    def set_indexes(self):
        # Set dict to proxy key val lookup access
        self.indexes = dict()
        idxs = slices_from_partition(self.count, self.partition)
        for group, (start, stop) in zip(self.groups, idxs):
            self.indexes[group] = self.IndexObj(name=group, start=start,
                                           stop=stop, allocated=stop-start)
        # Set instance vars for attribute access
        for g, idx in self.indexes.items():
            setattr(self, g, idx)

    def update_spec(self, spec, val):
        if spec not in ["partition", "count"]:
            raise ValueError("arg[0] must be one of {partition, count}"))
        setattr(self, spec, val)
        self.set_indexes()


class BaseVariableArray(ABCVariableIterator):
    _vtyp = "BaseVariableArray"

    def __init__(self, vset, arr):
        self._vset = vset
        # TODO Check arr iterable/iterator container-like w append(write)
        self._arr = arr

    @property
    def _arr(self):
        return self._arr_

    @_arr.setter
    def _arr(self, arr):
        self._arr_ = iter(arr)


class BaseSampleArray(ABCSampleIterator):
    _vtyp = "BaseSampleArray"
    _delegated = ["mltype", "classes", "image_shape", "image_dytpe"]

    def __init__(self, vset, group):
        self._vset = vset
        self._dgroup = group

    def __getattr__(self, name):
        if name in self._delegated:
            return self._vset.__getattribute__(name)
        raise AttributeError("this no have {}".format(name))

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


@delegate_metadata_props()
class BaseDataSet(ABCDataSet):
    _vtyp = "BaseDataSet"
    _groups = DATAGROUPS
    _mdprops = VSPropMap
    _fetch_class = NotImplemented
    _sample_class = NotImplemented
    _variable_class = NotImplemented

    def __init__(self, **kwargs):
        # Define a call signature here using inspect TODO
        self._vim_ = None
        self._train = None
        self._test = None
        self._valiate = None
        self._img_arr_ = None
        self._lbl_arr_ = None
        self._configure_instance()

        for k, v in kwargs.items():
            if k not in self._mdprops:
                raise ValueError("Unexpected initialization argument!")
            self._set_dprop(self, k, v)

    @classmethod
    def _set_dprop(cls, obj, dname, dval):
        try:
            p = getattr(cls, dname)
            p.__set__(obj, dval)
        except Exception as e:
            if val is not None:
                raise

    def _configure_instance(self):
        # do things like register callbacks etc here
        pass

    @property
    def _vim(self):
        if self._vim_ is None:
            try:
                prt = self.partition
                cnt = self.count
                grps = self._groups
            except AttributeError as ae:
                raise AttributeError("Data attrs {count, partition, _groups} must be set to access VIM") from None
            vim = VirtualIndexManager(prt, cnt, grps)
            self._mdhooks.partition.register(vim.update_spec)
            self._mdhooks.count.register(vim.update_spec)
            self._vim_ = vim
        return self._vim_

    @property
    def _img_handler_fn(self):
        return bytes_to_array

    @property
    def _lbl_handler_fn(self):
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
                     in self._mdprops.keys()])

