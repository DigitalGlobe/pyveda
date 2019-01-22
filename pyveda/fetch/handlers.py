import numpy as np
from shapely.ops import transform
import os
from tempfile import NamedTemporaryFile
from shapely.geometry import shape, box
from pyveda.utils import from_bounds
import numpy as np
from skimage.draw import polygon
from skimage.io import imread


def _on_fail(shape=(3, 256, 256), dtype=np.uint8):
    return np.zeros(shape, dtype=dtype)

def bytes_to_array(bstring):
    if bstring is None:
        return on_fail()
    try:
        fd = NamedTemporaryFile(prefix='veda', suffix='.tif', delete=False)
        fd.file.write(bstring)
        fd.file.flush()
        fd.close()
        arr = imread(fd.name)
        if len(arr.shape) == 3:
            arr = np.rollaxis(arr, 2, 0)
        else:
            arr = np.expand_dims(arr, axis=0)
    except Exception as e:
        arr = _on_fail()
    finally:
        fd.close()
        os.remove(fd.name)
    return arr


class BaseLabelHandler(object):
    def __init__(self, vset):
        self.vset = vset

    @staticmethod
    def _get_transform(bounds, height, width):
        return from_bounds(*bounds, width, height)

    def _parse_response(self, res):
        return res['properties']['label']

    def _payload_handler(self, *args, **kwargs):
        raise NotImplementedError


class ClassificationHandler(BaseLabelHandler):
    _default_dtype = np.uint8

    def _payload_handler(self, item, **kwargs):
        payload = self._parse_response(item)
        return [payload[klass] for klass in self.vset.classes]


class SegmentationHandler(BaseLabelHandler):
    _default_dtype = np.float32

    def _payload_handler(self, *args, **kwargs):
        return self._handle_pixel_payload(*args, **kwargs)

    def _handle_pixel_payload(self, item, **kwargs):
        payload = self._parse_response(item)
        out_shape = self.vset.image_shape
        if len(out_shape) == 3:
            out_shape = out_shape[-2:]
        out_array = np.zeros(out_shape)
        value = 1
        for klass in self.vset.classes:
            shapes = payload[klass]
            try:
                out_array += rasterize(((shape(g), value) for g in shapes), out_shape=out_shape)
            except Exception as e:
                pass
        return out_array

    @staticmethod
    def _handle_geo_payload(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        out_array = np.zeros(out_shape)
        value = 1
        for k, features in item['data']['label'].items():
            try:
                out_array += SegmentationHandler._create_mask(features, value, out_shape)
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


class ObjDetectionHandler(BaseLabelHandler):
    _default_dtype = np.float32

    def _payload_handler(self, *args, **kwargs):
        return self._handle_pixel_payload(*args, **kwargs)

    def _handle_pixel_payload(self, item, **kwargs):
        payload = self._parse_response(item)
        return [payload[klass] for klass in self.vset.classes]

    @staticmethod
    def _handle_geo_payload(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        xfm = BaseLabelHandler._get_transform(item['data']['bounds'], *out_shape)
        labels = []
        for k, features in item['data']['label'].items():
            class_labels = []
            for i, f in enumerate(features):
                b = shape(f).bounds
                ll, ur = ~xfm * (b[0],b[1]), ~xfm * (b[2],b[3])
                class_labels.append([*ll, *ur])
            labels.append(class_labels)
        return labels

def get_label_handler(vset):
    if vset.mltype == "classification":
        handler = ClassificationHandler(vset)
    elif vset.mltype == "segmentation":
        handler = SegmentationHandler(vset)
    else:
        handler = ObjDetectionHandler(vset)

    def fn(*args, **kwargs):
        return handler._payload_handler(*args, **kwargs)

    return fn
