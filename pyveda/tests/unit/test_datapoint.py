'''
Tests for DataPoint that don't rely on the server
'''
# path hack for veda installs
import os, sys
sys.path.append("..")
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
        self.assertTrue(isinstance(dp, DataPoint))
        assertEqual(dp.id, 'ae91f7df-ae37-4d31-9506-d9176f50403c')
        assertEqual(dp.mltype, 'classification')
        assertEqual(dp.dtype, 'uint8') # based on init logic and sample json
        assertEqual(dp.label, {"building": 0})
        assertEqual(dp.bounds. [
            -97.74107094008983,
            30.270496899310096,
            -97.74029824874955,
            30.271269590650377
        ])
        assertEqual(dp.tile_coords, ["962", "179"])
        assertEqual(type(shape(dp)), Polygon)
        assertEqual(dp.data['dataset_id'] == 'e91fb673-4a31-4221-a8ef-01706b6d9b63')

    def test_datapoint_fetch(self):
        #vc = VedaCollection('fake')
        vc = VedaCollection.from_id('e91fb673-4a31-4221-a8ef-01706b6d9b63')
        dp = vc.fetch('ae91f7df-ae37-4d31-9506-d9176f50403c')
        self.assertTrue(isinstance(dp, DataPoint))
        assertEqual(dp.id, 'ae91f7df-ae37-4d31-9506-d9176f50403c')
        assertEqual(dp.mltype, vc.mltype)
        self.assertTrue(isinstance(shape(vc), Polygon))
        assertEqual(dp.data['dataset_id'], 'e91fb673-4a31-4221-a8ef-01706b6d9b63')
        assertEqual(dp.meta['parent'], 'e91fb673-4a31-4221-a8ef-01706b6d9b63')
        assertEqual(dp.dtype, vc.dtype) # should inherit
