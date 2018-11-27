'''
Tests for DataPoint that don't rely on the server
'''
import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import DataPoint, VedaCollection
import json
from shapely.geometry import shape
from shapely.geometry.polygon import Polygon
import unittest

test_dir = os.path.dirname(__file__)
test_json = os.path.join(test_dir, 'responses', 'datapoint.json')


class DataPointTest(unittest.TestCase):
    def setUp(self):
        with open(test_json) as source:
            self.json = json.load(source)

    def test_datapoint(self):
        dp = DataPoint(self.json)
        print(dp)
        print(*dp.bounds)
        print(dp.__geo_interface__)
        self.assertTrue(isinstance(dp, DataPoint))
        self.assertEqual(dp.id, 'ae91f7df-ae37-4d31-9506-d9176f50403c')
        self.assertEqual(dp.mltype, 'classification')
        self.assertEqual(dp.dtype, 'uint8') # based on init logic and sample json
        self.assertEqual(dp.label, {"building": 0})
        self.assertEqual(type(shape(dp)), Polygon)
        self.assertEqual(dp.dataset_id, 'e91fb673-4a31-4221-a8ef-01706b6d9b63')

    def test_datapoint_fetch(self):
        #vc = VedaCollection('fake')
        vc_id = '82553d2f-9c9c-46f0-ad9f-1a27a8673637'
        dp_id = 'c5942231-dd6d-4ab8-9fce-04d28aa560d8'
        vc = VedaCollection.from_id(vc_id)
        dp = vc.fetch(dp_id)
        self.assertTrue(isinstance(dp, DataPoint))
        # public properties
        self.assertEqual(vc.id, vc_id)
        self.assertEqual(dp.mltype, vc.mltype)
        self.assertEqual(dp.dtype, vc.dtype) # should inherit
        self.assertEqual(dp.dataset_id, vc_id)
        self.assertEqual(dp.imshape, vc.imshape)
        self.assertEqual(dp.tile_coords, [964, 181])
        # geo interface
        self.assertEqual(dp.bounds, [
            -97.73952555740927,
            30.26895151662954,
            -97.73875286606899,
            30.26972420796982
        ])
        self.assertTrue(isinstance(shape(vc), Polygon))
