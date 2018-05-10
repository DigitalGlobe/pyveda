import os
import tables

MLTYPES = {"TRAIN": "Data designated for model training",
           "TEST": "Data designated for model testing",
           "VALIDATION": "Data designated for model validation"
          }

FRAMEWORKS = ["TensorFlow", "PyTorch", "Keras"]

fname = "/Users/jamiepolackwich1/projects/veda-nbs/test.h5"
dcmap = {"buildings": 1, "cars": 2, "zebras": 3}

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
        if isinstance(spec, slice):
            for rec in self.images.__iter__(spec)
                yield self._input_fn(rec)
        else:
            for rec in self._node.images[spec]:
                yield self._input_fn(rec)


    def __len__(self):
        return len(self._node.images)

class ImageTrainerH5(object):
    def __init__(self, fname=fname, cmap=dcmap, groups=MLTYPES, framework=None):
        self._framework = framework
        self._fw_loader = lambda x: x
        self._fileh = tables.open_file(fname, mode="w", title="whatever")
        Classifications = {klass: tables.UInt8Col(pos=idx + 2) for idx, (klass, _)
                               in enumerate(sorted(cmap.items(), key=lambda x: x[1]))}
        Classifications["image_chunk_index"] = tables.UInt8Col(shape=2, pos=1)
        # create data groups
        for name, desc in groups:
            self._fileh.create_group("/", name.lower(), desc)
        self._groups = {group._v_name: group for group in self._fileh.root._f_iter_nodes("Group")}
        # assign arrays
        #self._fileh.create_carray(self._g_train, "images")
        for name, group in self._groups.items():
            self._fileh.create_table(group, "hit_table", Classifications,
                                     "Chip Index + Klass Hit Record", tables.Filters(0))
            self._fileh.create_carray(group, "images", atom=tables.UInt8Atom(), shape=(3, 256, 256))
            self._fileh.create_group(group, "labels", "Image label type array data")
            self._fileh.create_carray("/{}/labels".format(name), "segmentations",
                                      atom=tables.UInt8Atom(), shape=(256, 256))
            self._fileh.create_vlarray("/{}/labels", "detections",
                                      atom=tables.UInt8Atom())

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







