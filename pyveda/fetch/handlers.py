import numpy as np
from shapely.ops import transform
import os
from tempfile import NamedTemporaryFile
from shapely.geometry import shape, box
from pyveda.utils import from_bounds
import numpy as np

from skimage.draw import polygon
from skimage.io import imread


class NDImageHandler(object):
    _default_dtype = np.float32

    @staticmethod
    def _payload_handler(*args, **kwargs):
        return NDImageHandler._bytes_to_array(*args, **kwargs)

    @staticmethod
    def _on_fail(shape=(3, 256, 256), dtype=np.uint8):
        return np.zeros(shape, dtype=dtype)

    @staticmethod
    def _bytes_to_array(bstring):
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
            arr = NDImageHandler._on_fail()
        finally:
            fd.close()
            os.remove(fd.name)
        return arr


class BaseLabelHandler(object):
    @staticmethod
    def _get_transform(bounds, height, width):
        return from_bounds(*bounds, width, height)

    @staticmethod
    def _parse_response(res):
        return res['properties']['label']

    @staticmethod
    def _payload_handler(*args, **kwargs):
        raise NotImplementedError


class ClassificationHandler(BaseLabelHandler):
    _default_dtype = np.uint8

    @staticmethod
    def _payload_handler(item, klasses=[], **kwargs):
        payload = ClassificationHandler._parse_response(item)
        return [payload[klass] for klass in klasses]


class SegmentationHandler(BaseLabelHandler):
    _default_dtype = np.float32

    @staticmethod
    def _payload_handler(*args, **kwargs):
        return SegmentationHandler._handle_pixel_payload(*args, **kwargs)

    @staticmethod
    def _handle_pixel_payload(item, klasses=[], out_shape=None, **kwargs):
        payload = SegmentationHandler._parse_response(item)
        if len(out_shape) == 3:
            out_shape = out_shape[-2:]
        out_array = np.zeros(out_shape, dtype=np.uint8)
        for value, klass in enumerate(klasses):
            value += 1
            shapes = payload[klass]
            if shapes:
                try:
                    out_array = SegmentationHandler._create_mask(
                        shapes, value, out_array)
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
                out_array = SegmentationHandler._create_mask(
                    features, value, out_array)
                value += 1
            except TypeError as e:
                pass
        return out_array

    @staticmethod
    def _create_mask(shapes, value, mask):
        def _apply(mask, coords):
            c, r = zip(*[(x, y) for x, y in coords])
            rr, cc = polygon(np.array(r), np.array(c))
            mask[rr, cc] = value
            return mask

        for f in shapes:
            if f['type'] == 'MultiPolygon':
                for coords in f['coordinates']:
                    mask = _apply(mask, coords[0])
            else:
                coords = f['coordinates'][0]
                mask = _apply(mask, coords)
        return mask


class ObjDetectionHandler(BaseLabelHandler):
    _default_dtype = np.float32

    @staticmethod
    def _payload_handler(*args, **kwargs):
        return ObjDetectionHandler._handle_pixel_payload(*args, **kwargs)

    @staticmethod
    def _handle_pixel_payload(item, klasses=[], out_shape=None, **kwargs):
        payload = ObjDetectionHandler._parse_response(item)
        return [payload[klass] for klass in klasses]

    @staticmethod
    def _handle_geo_payload(item, imshape):
        out_shape = imshape
        if len(imshape) == 3:
            out_shape = imshape[-2:]
        xfm = BaseLabelHandler._get_transform(
            item['data']['bounds'], *out_shape)
        labels = []
        for k, features in item['data']['label'].items():
            class_labels = []
            for i, f in enumerate(features):
                b = shape(f).bounds
                ll, ur = ~xfm * (b[0], b[1]), ~xfm * (b[2], b[3])
                class_labels.append([*ll, *ur])
            labels.append(class_labels)
        return labels
