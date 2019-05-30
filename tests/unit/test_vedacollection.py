'''
Tests for VedaCollection that don't rely on the server
'''
import os, sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

from pyveda.vedaset.veda.api import DataSampleClient, VedaCollectionProxy
import json
from shapely.geometry import shape, box
from shapely.geometry.polygon import Polygon
import unittest
from unittest import skip

test_dir = os.path.dirname(__file__)
test_json = os.path.join(test_dir, 'responses', 'vc.json')

# a valid vedacollection ID
VC_ID = '67a16de1-7baf-44bf-a779-2bf97a37c3bd'
# a valid datapoint from the above
DP_ID = '7f30b1ef-1622-41ca-ab21-9b66d23d87fc'

class VedaCollectionTest(unittest.TestCase):
    def setUp(self):
        with open(test_json) as source:
            self.json = json.load(source)

    def test_vedacollection(self):
        vc = VedaCollectionProxy.from_doc(self.json)
        self.assertTrue(isinstance(vc, VedaCollectionProxy))
        self.assertEqual(vc.mltype, 'classification')
        self.assertEqual(vc.percent_cached, 100)
        self.assertEqual(vc.name,'Austin Buildings Classification')
        self.assertEqual(vc.dtype, 'int8')
        self.assertEqual(vc.count, 250)
        self.assertEqual(vc.__geo_interface__, box(*vc.bounds).__geo_interface__)


class VCFetchTest(unittest.TestCase):

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch.yaml', filter_headers=['authorization'])
    def setUp(self):
        self.vc = pv.from_id(VC_ID)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_index.yaml', filter_headers=['authorization'])
    def test_fetch_index(self):
        dp = self.vc.fetch_sample_from_index(0)
        self.assertTrue(isinstance(dp, DataSampleClient))

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_slice.yaml', filter_headers=['authorization'])
    def test_fetch_points(self):
        dps = self.vc.fetch_samples_from_slice(5, num_points=5)
        self.assertEqual(len(dps), 5)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_sample_id.yaml', filter_headers=['authorization'])
    def test_fetch_ids(self):
        dp = self.vc.fetch_sample_from_id(DP_ID)
        self.assertTrue(isinstance(dp, DataSampleClient))

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_ids.yaml', filter_headers=['authorization'])
    def test_ids(self):
        ids = self.vc.gen_sample_ids(count=50, page_size=50, get_urls=False)
        nid = next(ids)
        self.assertTrue(isinstance(nid, str))
        self.assertEqual(len([x for x in ids]), 49)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_ids_urls.yaml', filter_headers=['authorization'])
    def test_ids_urls(self):
        ids = self.vc.gen_sample_ids(size=50, get_urls=True)
        nid = next(ids)
        self.assertTrue(isinstance(nid, tuple))
        self.assertEqual(len(nid), 3)
        assert nid[0].startswith('https://')
        assert nid[1].startswith('https://')


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_getatr.yaml', filter_headers=['authorization'])
    def test_getitem(self):
        dp = self.vc[0]
        self.assertTrue(isinstance(dp, DataSampleClient))
        dps = self.vc[0:3]
        self.assertTrue(isinstance(dps[0], DataSampleClient))
        self.assertEqual(len(dps), 2)

