''' Tests for functions in veda.api.py '''
from pyveda.vedaset.veda.api import VedaCollectionProxy
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
        ''' Test the append_from_geojson method '''
        type = np.dtype('uint8')
        vcp = VedaCollectionProxy(dataset_id='1234', dtype=type,
                                  status='COMPLETE', sensors=['WV4'])
        geojson = {}
        class _image():
            def __init__(self):
                self.dtype = type
        test_image = _image()
        vcp.append_from_geojson(geojson=geojson, image=test_image,
                                background_ratio=1.0)
        from_geo.assert_called_once_with(geojson, test_image,
                                        background_ratio=1.0, sensors=['WV4'],
                                        dtype=type, dataset_id='1234')
