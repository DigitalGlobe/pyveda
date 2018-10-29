import numpy as np
from rasterio.transform import from_bounds
from rasterio.features import rasterize
from shapely.ops import transform
from shapely.geometry import shape, box


class BaseLabel(object):
    def __init__(self, imshape):
        self.imshape = imshape

    @staticmethod
    def _get_transform(bounds, height, width):
        return from_bounds(*bounds, width, height)


class ClassificationLabel(BaseLabel):
    _default_dtype = np.uint8


class SegmentationLabel(BaseLabel):
    _default_dtype = np.float32

    @staticmethod
    def from_geo(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        xfm = BaseLabel._get_transform(item['data']['bounds'], *out_shape)
        out_array = np.zeros(out_shape)
        value = 1
        for k, features in item['data']['label'].items():
            try:
                out_array += SegmentationLabel._create_mask(features, value, out_shape, xfm)
                value += 1
            except Exception as e: # I think this is ValueError from rasterio but need check
                pass
        return out_array

    @staticmethod
    def _create_mask(shapes, value, _shape, tfm):
        return rasterize(((shape(g), value) for g in shapes), out_shape=_shape, transform=tfm)


class ObjDetectionLabel(BaseLabel):
    _default_dtype = np.float32

    @staticmethod
    def from_geo(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        xfm = BaseLabel._get_transform(item['data']['bounds'], *out_shape)
        labels = []
        for k, features in item['data']['label'].items():
            class_labels = []
            for f in features:
                b = _shape(f).bounds
                ll, ur = ~xfm * (b[0],b[1]), ~xfm * (b[2],b[3])
                class_labels.append([*ll, *ur])
            labels.append(class_labels)
        return np.array(labels)

