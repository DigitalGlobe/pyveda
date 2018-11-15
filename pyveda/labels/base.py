import numpy as np
from shapely.ops import transform
from shapely.geometry import shape, box
from pyveda.utils import from_bounds
import numpy as np
from skimage.draw import polygon


class BaseLabel(object):
    def __init__(self, imshape):
        self.imshape = imshape

    @staticmethod
    def _get_transform(bounds, height, width):
        return from_bounds(*bounds, width, height)

    @staticmethod
    def _parse_response(res):
        return res['properties']['label']

class ClassificationLabel(BaseLabel):
    _default_dtype = np.uint8

    @staticmethod
    def from_pixels(item, klasses=[], **kwargs):
        payload = ClassificationLabel._parse_response(item)
        return [payload[klass] for klass in klasses]


class SegmentationLabel(BaseLabel):
    _default_dtype = np.float32

    @staticmethod
    def from_pixels(item, klasses=[], out_shape=None, **kwargs):
        payload = SegmentationLabel._parse_response(item)
        if len(out_shape) == 3:
            out_shape = out_shape[-2:]
        out_array = np.zeros(out_shape)
        value = 1
        for klass in klasses:
            shapes = payload[klass]
            try:
                out_array += rasterize(((shape(g), value) for g in shapes), out_shape=out_shape)
            except Exception as e:
                pass
        return out_array

    @staticmethod
    def from_geo(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        out_array = np.zeros(out_shape)
        value = 1
        for k, features in item['data']['label'].items():
            try:
                out_array += SegmentationLabel._create_mask(features, value, out_shape)
                value += 1
            except Exception as e: # I think this is ValueError from rasterio but need check
                pass
        return out_array

    @staticmethod
    def _create_mask(shapes, value, _shape):
        mask = np.zeros(_shape, dtype=np.uint8)
        for f in shapes:
            coords = f['coordinates'][0]
            r, c = zip(*[(x,y) for x,y in coords])
            rr, cc = polygon(np.array(r), np.array(c))
            mask[rr, cc] = value


class ObjDetectionLabel(BaseLabel):
    _default_dtype = np.float32

    @staticmethod
    def from_pixels(item, klasses=[], out_shape=None, **kwargs):
        payload = ObjDetectionLabel._parse_response(item)
        return [payload[klass] for klass in klasses]

    @staticmethod
    def from_geo(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        xfm = BaseLabel._get_transform(item['data']['bounds'], *out_shape)
        labels = []
        for k, features in item['data']['label'].items():
            class_labels = []
            for i, f in enumerate(features):
                b = shape(f).bounds
                ll, ur = ~xfm * (b[0],b[1]), ~xfm * (b[2],b[3])
                class_labels.append([*ll, *ur])
            labels.append(class_labels)
        return labels

