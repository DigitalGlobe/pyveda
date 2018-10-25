import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.ops import transform
from shapely.geometry import shape, box


class BaseLabel(object):
    def __init__(self, imshape):
        self.imshape = imshape

    def _get_transform(self, bounds, height, width):
        return from_bounds(*bounds, width, height)


class ClassificationLabel(BaseLabel):
    _default_dtype = np.uint8


class SegmentationLabel(BaseLabel):
    _default_dtype = np.float32

    def _from_geo(self, item):
        out_shape = self.imshape
        xfm = self._get_transform(item['data']['bounds'], *out_shape)
        out_array = np.zeros(out_shape)
        value = 1
        for k, features in item['data']['label'].items():
            out_array += self._create_mask(features, value, out_shape, xfm)
            value += 1
        return out_array

    def _create_mask(self, shapes, value, _shape, tfm):
        return rasterize(((shape(g), value) for g in shapes), out_shape=_shape, transform=tfm)


class ObjDetectionLabel(BaseLabel):
    _default_dtype = np.float32

    def _from_geo(self, item):
        out_shape = self.imshape
        xfm = self._get_transform(item['data']['bounds'], *out_shape)
        labels = []
        for k, features in item['data']['label'].items():
            class_labels = []
            for f in features:
                b = shape(f).bounds
                ll, ur = ~xfm * (b[0],b[1]), ~xfm * (b[2],b[3])
                class_labels.append([*ll, *ur])
            labels.append(class_labels)
        return np.array(labels)

