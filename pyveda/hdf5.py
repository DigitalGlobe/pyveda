import os
import tables

MLTYPES = {"TRAIN": "Data designated for model training",
           "TEST": "Data designated for model testing",
           "VALIDATION": "Data designated for model validation"
          }

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

klass_map = {"buildings": 1, "cars": 2, "zebras": 3}

class LabelNotSupported(NotImplementedError):
    pass

class FrameworkNotSupported(NotImplementedError):
    pass

class WrappedDataArray(object):
    def __init__(self, node, trainer, input_fn=lambda x: x):
        self._arr = node
        self._trainer = trainer
        self._input_fn = input_fn

    def __iter__(self, spec=slice(None)):
        if isinstance(spec, slice):
            for rec in self._arr.iterrows(spec.start, spec.stop, spec.step):
                yield self._input_fn(rec)
        else:
            for rec in self._arr.image[spec]:
                yield self._input_fn(rec)

    def __getitem__(self, idx):
        return self._arr[idx]

    def __setitem__(self, key, value):
        raise NotSupportedException("For your protection, overwriting raw data in ImageTrainer is not supported.")

    def __len__(self):
        return len(self._arr)

    def append(self, item):
        dims = item.shape
        self._arr.append(item.reshape(1, *dims))

class WrappedDataNode(WrappedDataArray):
    def __init__(self, node, trainer):
        self._node = node
        self._trainer = trainer

    @property
    def image(self):
        return WrappedDataArray(self._node.image, self._trainer, input_fn = self._trainer._fw_loader)

    @property
    def classification(self):
        return WrappedDataArray(self._node.classification, self._trainer)

    @property
    def segmentation(self):
        return WrappedDataArray(self._node.labels.segmentation, self._trainer)

    @property
    def detection(self):
        return WrappedDataArray(self._node.labels.detection, self._trainer)

    def __getitem__(self, idx):
        label_data = getattr(self, self._trainer.focus)
        return list(zip(self.image[idx], label_data[idx]))

    def __iter__(self, spec=slice(None)):
        data = [getattr(self, label) for label in self._trainer.focus]
        data.insert(0, self.image)
        if isinstance(spec, slice):
            for rec in zip([arr[spec] for arr in data]):
                yield rec
        else:
            for rec in zip([arr[spec] for arr in data]):
                yield rec

    def __len__(self):
        return len(self._node.image)


class ImageTrainer(object):
    """
    An interface for consuming and reading local data intended to be used with machine learning training
    """
    def __init__(self, fname="test.h5", klass_map=klass_map, data_groups=MLTYPES, framework=None,
                 title="Unknown", image_shape=(3, 256, 256), focus="classification"):
        self._framework = framework
        self._fw_loader = lambda x: x
        imshape = list(image_shape)
        imshape.insert(0,0)
        self._imshape = tuple(imshape)
        self._segshape = tuple([s if idx > 0 else 0 for idx, s in enumerate(image_shape)])
        self.imshape = image_shape
        self._focus = focus
        self.klass_map = klass_map
        if not os.path.exists(fname):
            self._fileh = tables.open_file(fname, mode="w", title=title)
            for name, desc in data_groups.items():
                self._fileh.create_group("/", name.lower(), desc)

            Classifications = {klass: tables.UInt8Col(pos=idx + 2) for idx, (klass, _)
                                in enumerate(sorted(data_groups.items(), key=lambda x: x[1]))}
            Classifications["image_chunk_index"] = tables.UInt8Col(shape=2, pos=1)

            groups = {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}
            for name, group in groups.items():
                labels = self._fileh.create_group(group, "labels", "Image Labels")
                self._fileh.create_table(group, "hit_table", Classifications,
                                        "Chip Index + Klass Hit Record", tables.Filters(0))
                self._fileh.create_earray(group, "image", atom=tables.UInt8Atom(), shape=self._imshape)
                self._fileh.create_earray(group, "classification", atom=tables.UInt8Atom(), shape=(0, 1, len(klass_map)))
                self._fileh.create_earray(labels, "segmentation",
                                        atom=tables.UInt8Atom(), shape=self._segshape)
                self._fileh.create_vlarray(labels, "detection",
                                        atom=tables.UInt8Atom())

        else:
            self._fileh = tables.open_file(fname, mode="a", title=title)

    @property
    def focus(self):
        return self._focus
#        return [task for task, leaf in self._fileh.root.train.labels._v_children.items()
#                if isinstance(leaf, tables.array.Array)]

    @focus.setter
    def focus(self, foc):
        if foc not in ['classification', 'segmentation', 'detection']:
            raise LabelNotSupported("Focus must be classification, segmentation or detection")
        self._focus = foc

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
    def validation(self):
        return WrappedDataNode(self._fileh.root.validation, self)

    def __repr__(self):
        return self._fileh.__str__()
