''' Tests for VedaBase '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import VedaCollection
from pyveda.vedaset import VedaBase
from pyveda.fetch.handlers import ClassificationHandler, SegmentationHandler, ObjDetectionHandler
import json
from shapely.geometry import shape, box
from shapely.geometry.polygon import Polygon
from auth_mock import gbdx
import vcr
import unittest
from unittest import skip

def force(r1, r2):
    return True

my_vcr = vcr.VCR()
my_vcr.register_matcher('force', force)
my_vcr.match_on = ['force']

class VedaBaseLabelTest(unittest.TestCase):
    def test_classification(self):
        item = {
            'properties': {
                'label': {
                    'house':1, 'car':1, 'boat':0
                    }
            }
        }

        label = ClassificationHandler._payload_handler(item, klasses=['house','car'])
        self.assertEqual(label, [1,1])

    def test_segmentation(self):
        item = {
            "properties": {
                "label": {
                    "building": [
                        {
                            "type": "Polygon",
                            "coordinates": [
                                [
                                    [52, 84],
                                    [9, 195],
                                    [0, 102],
                                    [0, 69],
                                    [52, 84]
                                ]
                            ]
                        }
                    ]
                }
            }
        }
        label = SegmentationHandler._payload_handler(item, klasses=['building'], out_shape = [256,256])
        # Point inside is classified
        self.assertIs(label[1][80], 1)
        # Point on vertex is classified
        self.assertIs(label[52][84], 1)
        # Point outside is background value
        self.assertIs(label[256][256], 0)


    def test_object_detection(self):
        item = {
            "properties": {
                "label": {
                    "damaged building": [],
                    "building": [
                        [
                            235,
                            62,
                            256,
                            117
                        ]
                    ]
                }
            }
        }
        label = ObjDetectionHandler._payload_handler(item, klasses=['building', 'damaged building'], out_shape = [256,256])
        self.assertEqual(label[1], [])
        self.assertEqual(label[0][0], [235,62,256,117])

@skip('wait until all handlers pass')
class VedaBaseTest(unittest.TestCase):
    @my_vcr.use_cassette('tests/unit/cassettes/test_vedabase_setup.yaml', filter_headers=['authorization'])
    def setUp(self):
        vc_id = '82553d2f-9c9c-46f0-ad9f-1a27a8673637'
        dp_id = 'c5942231-dd6d-4ab8-9fce-04d28aa560d8'
        self.h5 = './test.h5'
        try:
            os.remove(self.h5)
        except OSError:
            pass
        self.vc = VedaCollection.from_id(vc_id)
        self.vb = self.vc.store(self.h5, size=10)

    def tearDown(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    def test_vedabase(self):
        self.assertTrue(isinstance(self.vb, VedaBase))
        # VBs support __len__ but maybe we should test these cases too
        #self.assertEqual(len(list(self.vb.train)), 7)
        #self.assertEqual(len(list(self.vb.train[:])), 7)
        self.assertEqual(len(self.vb.train), 7)
        self.assertEqual(self.vb.image_shape, self.vc.imshape)
        self.assertEqual(self.vb.mltype, self.vc.mltype)