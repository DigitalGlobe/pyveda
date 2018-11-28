'''
Tests for VedaCollection that don't rely on the server
'''
import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import DataPoint, VedaCollection, search
from auth_mock import gbdx
import json
from shapely.geometry import shape, box
from shapely.geometry.polygon import Polygon
import unittest
import vcr
from unittest import skip

test_dir = os.path.dirname(__file__)
test_json = os.path.join(test_dir, 'responses', 'vc.json')


class VedaCollectionTest(unittest.TestCase):
    def setUp(self):
        with open(test_json) as source:
            self.json = json.load(source)

    def test_vedacollection(self):
        vc = VedaCollection.from_doc(self.json)
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertEqual(vc.mltype, 'classification')
        self.assertEqual(vc.percent_cached, 100)
        self.assertEqual(vc.partition, [100, 0, 0])
        self.assertEqual(vc.name,'Austin Buildings Classification')
        self.assertEqual(vc.dtype, 'int8')
        self.assertEqual(vc.count, 250)
        self.assertEqual(vc.__geo_interface__, box(*vc.bounds).__geo_interface__)

    def test_vedacollection_from_id(self):
        vc = VedaCollection.from_id('82553d2f-9c9c-46f0-ad9f-1a27a8673637')
        self.assertTrue(isinstance(vc, VedaCollection))


# a valid vedacollection ID
VC_ID = '82553d2f-9c9c-46f0-ad9f-1a27a8673637'
# a valid datapoint from the above
DP_ID = 'c5942231-dd6d-4ab8-9fce-04d28aa560d8'

def force(r1, r2):
    return True

my_vcr = vcr.VCR()
my_vcr.register_matcher('force', force)
my_vcr.match_on = ['force']

class VedaCollectionTest_vcr(unittest.TestCase):
    def test_vedacollection_new(self):
        vc = VedaCollection('name')
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertEqual(vc.imshape, [0, 256, 256])
        self.assertEqual(vc.mltype, 'classification')
        self.assertEqual(vc.bounds, None)
        self.assertEqual(vc.count, 0)
      
    @my_vcr.use_cassette('tests/unit/cassettes/test_vc_search.yaml', filter_headers=['authorization'])
    def test_vc_search(self):
        results = search()
        vc = results[0]
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertGreater(vc.count, 0)
        self.assertEqual(type(shape(vc)), Polygon)
        self.assertEqual(vc.__geo_interface__, box(*vc.bounds).__geo_interface__)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vedacollection_id.yaml', filter_headers=['authorization'])
    def test_vedacollection_id(self):
        vc = VedaCollection.from_id(VC_ID)
        self.assertEqual(vc.id, VC_ID)
        self.assertEqual(vc.imshape, [3, 256, 256])
        self.assertEqual(vc.mltype,'classification')


class VCFetchTest(unittest.TestCase):

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch.yaml', filter_headers=['authorization'])
    def setUp(self):
        self.vc = VedaCollection.from_id(VC_ID)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_index.yaml', filter_headers=['authorization'])
    def test_fetch_index(self):
        dp = self.vc.fetch_index(0)
        self.assertTrue(isinstance(dp, DataPoint))


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_points.yaml', filter_headers=['authorization'])
    def test_fetch_points(self):
        dps = self.vc.fetch_points(5)
        self.assertEqual(len(dps), 5)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_points_index.yaml', filter_headers=['authorization'])
    def test_fetch_points_offset(self):
        dps = self.vc.fetch_points(10,5)
        self.assertEqual(len(dps), 10)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_ids.yaml', filter_headers=['authorization'])
    def test_fetch_ids(self):
        ids = self.vc.fetch_ids(page_size=50)
        self.assertEqual(len(ids[0]), 50)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_ids.yaml', filter_headers=['authorization'])
    def test_ids(self):
        ids = self.vc.ids(size=50, get_urls=False)
        nid = next(ids)
        print(nid)
        self.assertTrue(isinstance(nid, str))
        self.assertEqual(len(list(ids)), 49)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_ids_urls.yaml', filter_headers=['authorization'])
    def test_ids_urls(self):
        ids = self.vc.ids(size=50, get_urls=True)
        nid = next(ids)
        self.assertTrue(isinstance(nid, list))
        self.assertEqual(len(nid), 2)
        self.assertEqual(nid[0][:5], 'https')


    #@my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_getatr.yaml', filter_headers=['authorization'])
    def test_getitem(self):
        dp = self.vc[0]
        #self.assertTrue(isinstance(dp, DataPoint))
        dps = self.vc[0:2]
        self.assertTrue(isinstance(dps[0], DataPoint))
        self.assertEqual(len(dps), 2)

