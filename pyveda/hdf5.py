import os
import tables

MLTYPES = {"TRAIN": "Data designated for model training",
           "TEST": "Data designated for model testing",
           "VALIDATION": "Data designated for model validation"
          }

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

klass_map = {"buildings": 1, "cars": 2, "zebras": 3}

class FrameworkNotSupported(NotImplementedError):
    pass

class WrappedDataArray(object):
    def __init__(self, node, trainer):
        self._arr = node
        self._trainer = trainer

    def __iter__(self, spec=slice(None)):
        if isinstance(spec, slice):
            for rec in self._arr.iterrows(spec.start, spec.stop, spec.step):
                yield self._input_fn(rec)
        else:
            for rec in self._arr.images[spec]:
                yield self._input_fn(rec)

    def __setitem__(self, key, value):
        raise NotSupportedException("For your protection, overwriting raw data in ImageTrainer is not supported.")

    def __len__(self):
        return len(self._arr)

class WrappedDataNode(WrappedDataArray):
    def __init__(self, node, trainer):
        self._node = node
        self._trainer = trainer

    @property
    def images(self):
        return WrappedDataArray(self._node.images)

    @property
    def segmentations(self):
        return WrappedDataArray(self._node.labels.segmentations)

    @property
    def detections(self):
        return WrappedDataArray(self._node.labels.detections)

    def __iter__(self, spec=slice(None)):
        data = [getattr(self, label) for label in self._trainer.focus]
        data.insert(0, self.images).
        if isinstance(spec, slice):
            for rec in zip([arr.__iter__(spec) for arr in data]):
                yield rec
        else:
            for rec in zip([arr[spec] for arr in data]):
                yield rec

    def __len__(self):
        return len(self._node.images)

class ImageTrainerH5(object):
    def __init__(self, fname=fname, klass_map=klass_map, data_groups=MLTYPES, framework=None, title="Unknown"):
        self._framework = framework
        self._fw_loader = lambda x: x
        if not os.path.exists(fname):
            self._fileh = tables.open_file(fname, mode="w", title=title)
            for name, desc in data_groups:
                self._fileh.create_group("/", name.lower(), desc)

            Classifications = {klass: tables.UInt8Col(pos=idx + 2) for idx, (klass, _)
                                in enumerate(sorted(cmap.items(), key=lambda x: x[1]))}
            Classifications["image_chunk_index"] = tables.UInt8Col(shape=2, pos=1)

            groups = {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}
            for name, group in groups.items():
                self._fileh.create_table(group, "hit_table", Classifications,
                                        "Chip Index + Klass Hit Record", tables.Filters(0))
                self._fileh.create_earray(group, "data", atom=tables.UInt8Atom(), shape=(3, 256, 256))
                self._fileh.create_earray("/{}/labels".format(name), "segmentations",
                                        atom=tables.UInt8Atom(), shape=(256, 256))
                self._fileh.create_vlarray("/{}/labels", "detections",
                                        atom=tables.UInt8Atom())

        else:
            self._fileh = tables.open_file(fname, mode="a", title=title)

    @property
    def focus(self):
        if not self._focus:
            return self._focus
        return [task for task, leaf in self._fileh.root.images.labels._v_children.items()
                if isinstance(tables.array.Array)]

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
        return WrappedArrayNode(self._fileh.root.train)

    @property
    def test(self):
        return WrappedArrayNode(self._fileh.root.test)

    @property
    def validation(self):
        return WrappedArrayNode(self._fileh.root.validation)

