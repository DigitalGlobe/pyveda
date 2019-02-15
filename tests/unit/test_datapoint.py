import unittest
from shapely.geometry.polygon import Polygon
from shapely.geometry import shape
import json
from pyveda.veda.api import DataSampleClient
'''
Tests for DataPoint that don't rely on the server
'''
import os
import sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)


test_dir = os.path.dirname(__file__)
test_json = os.path.join(test_dir, 'responses', 'datapoint.json')


class DataSampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        with open(test_json) as source:
            self.json = json.load(source)

    def test_datapoint(self):
        dp = DataSampleClient(self.json)
        self.assertTrue(isinstance(dp, DataSampleClient))
        self.assertEqual(dp.id, 'ae91f7df-ae37-4d31-9506-d9176f50403c')
        self.assertEqual(dp.mltype, 'classification')
        self.assertEqual(dp.label, {"building": 0})
        self.assertEqual(type(shape(dp)), Polygon)
        self.assertEqual(dp.dataset_id, 'e91fb673-4a31-4221-a8ef-01706b6d9b63')

    @my_vcr.use_cassette(
        'tests/unit/cassettes/test_datapoint_fetch.yaml', filter_headers=['authorization'])
    def test_datapoint_fetch(self):
        vc_id = '67a16de1-7baf-44bf-a779-2bf97a37c3bd'
        dp_id = '7f30b1ef-1622-41ca-ab21-9b66d23d87fc'
        vc = pv.from_id(vc_id)
        dp = vc.fetch_sample_from_id(dp_id)
        self.assertTrue(isinstance(dp, DataSampleClient))
        # public properties
        self.assertEqual(vc.id, vc_id)
        self.assertEqual(dp.mltype, vc.mltype)
        self.assertEqual(dp.dtype, vc.dtype)  # should inherit
        self.assertEqual(dp.dataset_id, vc_id)
        self.assertEqual(dp.tile_coords, [965, 167])
        # geo interface
        self.assertEqual(dp.bounds, [
            -97.7503432361732,
            30.268178825289258,
            -97.74957054483292,
            30.26895151662954
        ])
        self.assertTrue(isinstance(shape(vc), Polygon))
