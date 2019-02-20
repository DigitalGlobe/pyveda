from collections import OrderedDict


class OpRegister(object):
    def __init__(self):
        self._ops_ = OrderedDict()

    def register(self, f, name=None, index=None):
        assert callable(f)
        if not name:
            name = getattr(f, "__name__", None) or f.__func__.__name__
        if index is None or index >= len(self._ops_):
            self._ops_[name] = f
        ops = list(self._ops_.items())
        ops.insert(index, (name, f))
        self._ops_ = OrderedDict(ops)

    def clear(self):
        self._ops_ = OrderedDict()

    @property
    def _ops(self):
        return self._ops_.values()

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._ops_.__getitem__(key)
        if isinstance(key, int):
            return list(self._ops_.values())[key]
        raise ValueError("Indexing supports str or int types only")

    def __iter__(self):
        return iter(self._ops)



class TransformRegister(OpRegister):
    def __init__(self, transforms=[], attach=False):
        super(TransformRegister, self).__init__()
        self._attached = attach
        for name, f in transforms:
            self.register(f, name)

    def attach(self):
        self._attached = True

    def detach(self):
        self._attached = False

    @property
    def _ops(self):
        if self._attached:
            return self._ops_.values()
        return []

    def transform(self, arg):
        _output = arg
        for fn in self:
            _output = fn(_output)
        return _output



class WrappedIterator(object):
    def __init__(self, arr):
        self._source = arr

    @property
    def _source(self):
        return self._source

    @_source.setter
    def _source(self, source):
        if not is_iterator(source):
            raise TypeError("Input source must be define iterator interface")
        self._source = source

    def __getitem__(self, obj):
        return self._source.__getitem__(obj)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._source)


