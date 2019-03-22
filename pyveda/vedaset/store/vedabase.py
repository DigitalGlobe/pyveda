import os
from functools import partial, wraps
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype, ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.vedaset.store.arrays import ClassificationArray, SegmentationArray, ObjDetectionArray, NDImageArray
from pyveda.vedaset.abstract import BaseSampleArray, BaseDataSet
from pyveda.frameworks.batch_generator import VedaStoreGenerator
from pyveda.vv.labelizer import Labelizer

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

MLTYPE_MAP = {"classification": ClassificationArray,
              "segmentation": SegmentationArray,
              "object_detection": ObjDetectionArray}

DATA_GROUPS = {"TRAIN": "Data designated for model training",
               "TEST": "Data designated for model testing",
               "VALIDATE": "Data designated for model validation"}

ignore_NaturalNameWarning = partial(ignore_warnings, _warning=tables.NaturalNameWarning)


class WrappedDataNode(object):
    def __init__(self, node, trainer):
        self._node = node
        self._vset = trainer

    @property
    def images(self):
        return self._vset._image_array_factory(self._node.images, self._vset, output_transform = self._vset._fw_loader)

    @property
    def labels(self):
        return self._vset._label_array_factory(self._node.hit_table, self._node.labels,  self._vset)

    def batch_generator(self, batch_size, steps=None, loop=True, shuffle=True, channels_last=False, expand_dims=False, rescale=False,
                        flip_horizontal=False, flip_vertical=False, label_transform=None,
                        batch_label_transform=None, image_transform=None, batch_image_transform=None, pad=None, **kwargs):
        """
        Generatates Batch of Images/Lables on a VedaBase partition.

        Args
            cache (VedaBase or VedaStream partition): Partition (train, test, or validate)
            batch_size (int): Number of samples in batch
            steps (int): Number of steps of batches to run in one epoch. If not provided, will calculate maximum possible number of complete batches
            loop (Boolean): Loop batcher indefinitely. If false, StopIteration is thrown after one epoch.
            shuffle (Boolean): Shuffle data between epochs.
            channels_last (Boolean): To return image data as Height-Width-Depth, instead of the default Depth-Height-Width
            rescale (Boolean): Return images rescaled to values between 0 and 1
            flip_horizontal (Boolean): Horizontally flip image and labels (50% probability)
            flip_vertical (Boolean): Vertically flip image and labels (50% probability)
            pad (int): Pad image with zeros to this dimension.
            label_transform (callable): function to apply to each label
            image_transform (callable): function to apply to each image
            batch_label_transform (callable): function to apply to the batch of labels
            batch_image_transform (callable): function to apply to the batch of images
        """
        return VedaStoreGenerator(self, batch_size=batch_size, steps=steps, loop=loop, shuffle=shuffle,
                                channels_last=channels_last, expand_dims = expand_dims, rescale=rescale,
                                flip_horizontal=flip_horizontal, flip_vertical=flip_vertical,
                                label_transform=label_transform,
                                batch_label_transform=batch_label_transform,
                                image_transform=image_transform,
                                batch_image_transform=batch_image_transform,
                                pad=pad, **kwargs)

    def __getitem__(self, spec):
        if isinstance(spec, int):
            return [self.images[spec], self.labels[spec]]
        else:
            return list(self.__iter__(spec))

    def __iter__(self, spec=None):
        if not spec:
            spec = slice(0, len(self), 1)
        gimg = self.images.__iter__(spec)
        glbl = self.labels.__iter__(spec)
        while True:
            yield (gimg.__next__(), glbl.__next__())

    def __len__(self):
        return len(self._node.images)

    def clean(self, count=None):
        """
        Page through VedaStream data and flag bad data.
        Params:
            count: the number of tiles to clean
        """
        classes = self._vset.classes
        mltype = self._vset.mltype
        Labelizer(self, mltype, count, classes).clean()


class H5DataBase(BaseDataSet):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname, mltype=None, klasses=None, image_shape=None,
                 image_dtype=None, title="NoTitle", framework=None,
                 overwrite=False, mode="a"):
        self._framework = framework
        self._fw_loader = lambda x: x

        if os.path.exists(fname):
            # TODO need to figure how to deal with existing files.
            if overwrite:
                os.remove(fname)
            else:
                self._load_existing(fname, mode)
                return

        self._fileh = tables.open_file(fname, mode="a", title=title)
        self._fileh.root._v_attrs.mltype = mltype
        self._fileh.root._v_attrs.klasses = klasses
        self._fileh.root._v_attrs.image_shape = image_shape
        self._fileh.root._v_attrs.image_dtype = image_dtype

        self._configure_instance()
        self._build_filetree()

    def _load_existing(self, fname, mode="a"):
        if mode == "w":
            raise ValueError("Opening the file in write mode will overwrite the file")
        self._fileh = tables.open_file(fname, mode=mode)
        self._configure_instance()

    def _configure_instance(self, *args, **kwargs):
        self._image_klass = NDImageArray
        self._label_klass = MLTYPE_MAP[self.mltype]
        self._classifications = dict([(klass, tables.UInt8Col(pos=idx + 1)) for idx, klass in enumerate(self.classes)])

    def _build_filetree(self, dg=DATA_GROUPS):
        # Build group nodes
        for name, desc in dg.items():
            self._fileh.create_group("/", name.lower(), desc)
        # Build table, array leaves
        self._create_tables(self._classifications, filters=tables.Filters(0))
        self._create_arrays(self._image_klass, self.image_dtype)
        self._create_arrays(self._label_klass)

    def _image_array_factory(self, *args, **kwargs):
        return self._image_klass(*args, **kwargs)

    def _label_array_factory(self, *args, **kwargs):
        return self._label_klass(*args, **kwargs)

    @ignore_NaturalNameWarning
    def _create_arrays(self, data_klass, data_dtype=None):
        for name, group in self._groups.items():
            data_klass.create_array(self, group, data_dtype)

    @ignore_NaturalNameWarning
    def _create_tables(self, classifications, filters=tables.Filters(0)):
        for name, group in self._groups.items():
            self._fileh.create_table(group, "hit_table", classifications,
                                     "Label Hit Record", filters)

    def _build_label_tables(self, rebuild=True):
        pass

    @property
    def mltype(self):
        return self._fileh.root._v_attrs.mltype

    @property
    def classes(self):
        return self._fileh.root._v_attrs.klasses

    @property
    def image_shape(self):
        return self._fileh.root._v_attrs.image_shape

    @property
    def image_dtype(self):
        return self._fileh.root._v_attrs.image_dtype

    @property
    def _groups(self):
        return {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}

    @property
    def framework(self):
        return self._framework

    @framework.setter
    def framework(self, fw):
        if fw and fw not in FRAMEWORKS:
            raise FrameworkNotSupported("Image adaptor not supported for {}".format(fw))
        self._framework = fw
        self._fw_loader = lambda x: x # TODO: Custom loaders here

    @property
    def train(self):
        return WrappedDataNode(self._fileh.root.train, self)

    @property
    def test(self):
        return WrappedDataNode(self._fileh.root.test, self)

    @property
    def validate(self):
        return WrappedDataNode(self._fileh.root.validate, self)

    def flush(self):
        self._fileh.flush()

    def close(self):
        self._fileh.close()

    def remove(self):
        raise NotImplementedError

    def __len__(self):
        return sum([len(self.train), len(self.test), len(self.validate)])

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        return self._fileh.__str__()

    def __del__(self):
        self.close()

    @classmethod
    def from_path(cls, fname, **kwargs):
        inst = cls(fname, **kwargs)
        return inst

    #@classmethod
    #def from_vc(cls, vc, **kwargs):
    #    # Load an empty H5DataBase from a VC
