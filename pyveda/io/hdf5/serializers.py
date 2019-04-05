import os
import tables
import numpy as np
import ujson as json
from functools import partial
from pyveda.vedaset.abstract import mltypes
from pyveda.io.utils import _atom_from_dtype
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported


class IOH5Interface(object):
    """
    Base interface for defining the serialization
    functions and methods that are used by H5Store
    classes.
    """

    @staticmethod
    def _input_fn(item):
        return item

    @staticmethod
    def _output_fn(item):
        return item

    @staticmethod
    def create_array(vset, group, arr_name=None):
        raise NotImplementedError


class IOImageArray(IOH5Interface):
    mltype = np.ndarray

    @staticmethod
    def _input_fn(item):
        dims = item.shape
        if len(dims) == 4:
            return item # for batch append stacked arrays
        if len(dims) in (2, 3): # extend single image array along axis 0
            return item.reshape(1, *dims)
        return item # what could this thing be, let it fail

    @staticmethod
    def create_array(vset):
        atom = _atom_from_dtype(vset.image_dtype)
        shape = list(vset.image_shape)
        shape.insert(0,0)
        shape = tuple(shape)
        vset._fileh.create_earray(vset._root,
                                  "images", atom=atom, shape=shape)


class IOBinaryClassification(IOH5Interface):
    mltype = "classification"

    @staticmethod
    def _input_fn(item):
        dims = item.shape
        if len(dims) == 2:
            return item # for batch append stacked arrays
        return item.reshape(1, *dims)

    @staticmethod
    def create_array(vset):
        atom = tables.UInt8Atom()
        shape = (0, len(vset.classes))
        vset._fileh.create_earray(vset._root,
                                  "labels", atom=atom, shape=shape)


class IOInstanceSegmentation(IOH5Interface):
    mltype = "segmentation"

    @staticmethod
    def create_array(vset):
        atom = _atom_from_dtype(np.int8)
        shape = tuple([s if idx > 0 else 0 for
                       idx, s in enumerate(vset.image_shape)])
        vset._fileh.create_earray(vset._root,
                                  "labels", atom=atom, shape=shape)


class IOObjectDetection(IOH5Interface):
    mltype = "obj_detection"

    @staticmethod
    def _input_fn(item):
        assert isinstance(item, list)
        return np.fromstring(json.dumps(item), dtype=np.uint8)

    @staticmethod
    def _output_fn(item):
        return json.loads(item.tostring())

    @staticmethod
    def append_batch(vset, items):
        for item in items:
            vset.append(item)

    @staticmethod
    def create_array(vset, filters=tables.Filters(complevel=0)):
        atom = tables.UInt8Atom()
        vset._fileh.create_vlarray(vset._root,
                                   "labels", atom=atom, filters=filters)


def adapt_serializers(vset):
    """
    Takes a HDF5-based DataStore object and sets
    DataStore._variable_class with the approprite
    serializer io functions based on the datatype
    of the store.
    """

    image_ios = (IOImageArray,)
    label_ios = (IOBinaryClassification,
                 IOInstanceSegmentation,
                 IOObjectDetection,)

    def get_serializer(ios):
        try:
            return [io for io in ios if
                    mltypes.types_match(vset, io)].pop()
        except IndexError:
            raise MLTypeError(
                "'{}' is not of type 'mltypes.mltype'".format(vset)) from None

    def adapt_serializer(io):
        adapted = partial(vset._variable_class,
                          input_fn = io._input_fn,
                          output_fn = io._output_fn)
        setattr(adapted, "create_array",
                partial(io.create_array, vset))
        return adapted

    img_io = list(image_ios).pop()
    img_class = adapt_serializer(img_io)
    setattr(vset, "_image_class_", img_class)

    lbl_io = get_serializer(label_ios)
    lbl_class = adapt_serializer(lbl_io)
    setattr(vset, "_label_class_", lbl_class)

    return vset


