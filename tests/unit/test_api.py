''' Tests for functions in veda.api.py '''
from pyveda.veda.api import VedaCollectionProxy
from unittest.mock import patch
import pyveda as pv
pv.config.set_dev()
import unittest
import numpy as np

class VedaAPIFunctionsTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @patch('pyveda.veda.api.VedaCollectionProxy.status')
    @patch('pyveda.veda.api.from_geo')
    def test_appendfromgeojson(self, from_geo, status):
        type = np.dtype('uint8')
        vcp = VedaCollectionProxy(dataset_id='1234', dtype=type, status='COMPLETE', sensors=[])
        geojson = {}
        class _image():
            def __init__(self):
                self.dtype = type

        mock_doc = {
            "name":             'mockname',
            "classes":          [],
            "dataset_id":       '234',
            "dtype":            np.dtype('uint8'),
            "userId":           'foo',
            "imshape":          [],
            "releases":         'releases',
            "mltype":           'classification',
            "tilesize":         [],
            "image_refs":       [],
            "sensors":          [],
            "bounds":           [],
            "public":           True,
            "count":            3,
            "percent_cached":   1,
            "background_ratio": 1.0
        }
        from_geo.return_value = {'properties': mock_doc}

        doc = vcp.append_from_geojson(geojson=geojson, image=_image(), background_ratio=1.0)
        self.assertEqual(mock_doc, doc['properties'])
