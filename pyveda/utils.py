import json
import datetime
import os
import uuid
import tempfile
import shutil
import h5py
import threading
import warnings
import inspect
import collections
from functools import partial
from shapely.geometry import box, mapping, shape
from shapely import ops
from affine import Affine


def tail(n, iterable):
    "Return an iterator over the last n items"
    # tail(3, 'ABCDEFG') --> E F G
    return iter(collections.deque(iterable, maxlen=n))


def check_unexpected_kwargs(kwargs, **unexpected):
    for key, message in unexpected.items():
        if key in kwargs:
            raise ValueError(message)


def parse_kwargs(kwargs, *name_and_values, **unexpected):
    values = [kwargs.pop(name, default_value)
              for name, default_value in name_and_values]
    if kwargs:
        check_unexpected_kwargs(kwargs, **unexpected)
        caller = inspect.stack()[1]
        args = ', '.join(repr(arg) for arg in sorted(kwargs.keys()))
        message = caller[3] + \
            '() got unexpected keyword argument(s) {}'.format(args)
        raise TypeError(message)
    return tuple(values)


def assert_kwargs_empty(kwargs):
    # It only checks if kwargs is empty.
    return parse_kwargs(kwargs)


def transform_to_int(tfm, x, y):
    return tuple(map(int, tfm * (x, y)))


def from_bounds(west, south, east, north, width, height):
    """Return an Affine transformation given bounds, width and height.
    Return an Affine transformation for a georeferenced raster given
    its bounds `west`, `south`, `east`, `north` and its `width` and
    `height` in number of pixels.

    Taken from Rasterio source:
        https://github.com/mapbox/rasterio/blob/master/rasterio/transform.py#L107
    """
    return Affine.translation(west, north) * Affine.scale(
        (east - west) / width, (south - north) / height)


def features_to_pixels(image, features, mltype):
    """
      Converts geometries to pixels coords for object detection and segmentation
      Each feature is converted using the bounds of the givein image.

      Args:
          image (rda image): The rda image to use for pixel transformations
          features (list): a list of geojson feature geometries
          mytype (str): the mltype of data that should be returned
    """
    if mltype == 'classification':
        return features
    else:
        _, size_y, size_x = image.shape
        params = image.bounds + (size_x, size_y)
        xfm = partial(transform_to_int, ~from_bounds(*params))
        converted = []
        for f in features:
            if mltype == 'object_detection':
                geom = box(*shape(f).bounds).intersection(shape(image))
                converted.append(list(map(int, ops.transform(xfm, geom).bounds)))
            elif mltype == 'segmentation':
                geom = shape(f).intersection(shape(image))
                converted.append(mapping(ops.transform(xfm, geom)))
        return converted


def mklogfilename(prefix, suffix="json", path=None):
    timestamp = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    basename = "_".join([prefix, timestamp]) # e.g. 'mylogfile_120508_171442'
    filename = ".".join([basename, suffix])
    if path:
        filename = os.path.join(path, filename)
    return filename


def mktempfilename(prefix, suffix="h5", path=None):
    idstr = str(uuid.uuid4())
    basename = "_".join([prefix, idstr])
    filename = ".".join([basename, suffix])
    if path:
        filename = os.path.join(path, filename)
    return filename


class NamedTemporaryHDF5File(object):
    def __init__(self, prefix="veda", suffix="h5", path=None, delete=True):
        self.name = mktempfilename(prefix, suffix=suffix, path=path)
        self._delete = delete
        self._fh = h5py.File(self.name, "a")

    def __enter__(self):
        return self._fh

    def __exit__(self, *args):
        if self._delete:
            self.remove()

    def remove(self):
        try:
            self._fh.close()
        except Exception:
            pass
        try:
            os.remove(self.name)
        except Exception:
            pass


class NamedTemporaryHDF5Generator(object):
    def __init__(self, dirpath=None, delete=True, delete_files=True):
        if not dirpath:
            dirpath = tempfile.mkdtemp()
        self.dirpath = dirpath
        self._delete = delete
        self._delete_files = delete_files
        self._fps = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self._delete_files:
            for h5 in self._fps:
                h5.remove()
        if self._delete:
            self.remove()

    def remove(self):
        for h5 in self._fps:
            h5.remove()
        shutil.rmtree(self.dirpath)

    def mktemp(self, prefix="veda", suffix="h5", delete=True):
        h5 = NamedTemporaryHDF5File(prefix=prefix, suffix=suffix,
                                                path=self.dirpath, delete=delete)
        self._fps.append(h5)
        return h5._fh, h5.name

    def mktempfilename(self, prefix="veda", suffix="h5"):
        return mktempfilename(prefix, suffix=suffix, path=self.dirpath)


def ignore_warnings(fn, _warning=None):
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            if not _warning:
                warnings.simplefilter("ignore")
            else:
                warnings.simplefilter("ignore", _warning)
            return fn(*args, **kwargs)
    return wrapper


def in_ipython_runtime_env():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True   # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return True   # Terminal running Ipython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Non-interactive runtime


class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stopper = threading.Event()

    def stop(self):
        self._stopper.set()

    def stopped(self):
        return self._stopper.is_set()



