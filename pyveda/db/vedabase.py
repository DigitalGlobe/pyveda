import os
from functools import partial, wraps
from collections import OrderedDict, defaultdict
import numpy as np
import tables
from pyveda.utils import mktempfilename, _atom_from_dtype, ignore_warnings
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from pyveda.db.arrays import ClassificationArray, SegmentationArray, ObjDetectionArray, ImageArray

from ipywidgets import interact
from IPython.display import Image, display
import ipywidgets as widgets
import numpy as np
from skimage.color import label2rgb
import matplotlib.pyplot as plt

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

MLTYPE_MAP = {"classification": ClassificationArray,
              "segmentation": SegmentationArray,
              "object_detection": ObjDetectionArray}

DATA_GROUPS = {"TRAIN": "Data designated for model training",
               "TEST": "Data designated for model testing",
               "VALIDATE": "Data designated for model validation"}

ignore_NaturalNameWarning = partial(ignore_warnings, _warning=tables.NaturalNameWarning)

def _format_image(image):
    if len(image.shape) == 2:
        return image
    elif len(image.shape) == 3:
        if image.shape[0] == 3:
            return np.rollaxis(image, 0, 3)
    raise ValueError("Image is not greyscale or 3-band")


class WrappedDataNode(object):
    def __init__(self, node, trainer):
        self._node = node
        self._trainer = trainer

    @property
    def images(self):
        return self._trainer._image_array_factory(self._node.images, self._trainer, output_transform = self._trainer._fw_loader)

    @property
    def labels(self):
        return self._trainer._label_array_factory(self._node.hit_table, self._node.labels,  self._trainer)

    def __getitem__(self, spec):
        if isinstance(spec, int):
            return [self.images[spec], self.labels[spec]]
        else:
            return list(self.__iter__(spec))

    def __iter__(self, spec=None):
        if not spec:
            spec = slice(0, len(self)-1, 1)
        gimg = self.images.__iter__(spec)
        glbl = self.labels.__iter__(spec)
        while True:
            yield (gimg.__next__(), glbl.__next__())

    def __len__(self):
        return len(self._node.images)

    def _plot_overlay(self, idx=0):
        image, label = self[idx]
        rgb = _format_image(image)
        image_label_overlay = label2rgb(label, image=rgb, bg_label=0)
        fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)
        axes[0].imshow(rgb, cmap=plt.cm.gray, interpolation='nearest')
        axes[1].imshow(image_label_overlay, interpolation='nearest')

        for a in axes:
            a.axis('off')
        plt.tight_layout()
        plt.show()

#    @interact(idx=widgets.IntSlider(min=0,max=100,step=1,value=0))
    def preview(self):
        interact(self._plot_overlay, idx=widgets.IntSlider(min=0, max=99, step=1, value=0))



class VedaBase(object):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname, mltype=None, klasses=None, image_shape=None, image_dtype=None, framework=None,
                 title="VedaBase", label_dtype=None, overwrite=False, mode="a"):

        self._framework = framework
        self._fw_loader = lambda x: x

        if os.path.exists(fname):
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
        self._image_klass = ImageArray
        self._label_klass = MLTYPE_MAP[self.mltype]
        self._classifications = dict([(klass, tables.UInt8Col(pos=idx + 1)) for idx, klass in enumerate(self.klasses)])

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
    def klasses(self):
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

    def __repr__(self):
        return self._fileh.__str__()

    def __del__(self):
        self.close()


