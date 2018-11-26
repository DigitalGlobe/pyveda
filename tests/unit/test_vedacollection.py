'''
Tests for VedaCollection that don't rely on the server
'''
import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import DataPoint, VedaCollection
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
        vc = VedaCollection(name = self.json['properties']['name'], dtype = self.json['properties']['dtype'],
                            percent_cached = self.json['properties']['percent_cached'],
                            count = self.json['properties']['count'],
                            bounds = self.json['properties']['bounds'])
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertEqual(vc.mltype, 'classification')
        self.assertEqual(vc.percent_cached, 100)
        self.assertEqual(vc.partition, [100, 0, 0])
        self.assertEqual(vc.name,'Austin Buildings Classification')
        self.assertEqual(vc.dtype, 'int8')
        self.assertEqual(vc.count, 250)
        self.assertEqual(vc.__geo_interface__, box(*vc.bounds).__geo_interface__)

    def test_vedacollection_from_id(self):
        vc = VedaCollection.from_id('e91fb673-4a31-4221-a8ef-01706b6d9b63')
        self.assertTrue(isinstance(vc, VedaCollection))

    # def test_vedacollection_points(self):
    #     vc = VedaCollection.from_id('e91fb673-4a31-4221-a8ef-01706b6d9b63')
    #     assertEqual(len(vc[0:250]), 250)
    #     assertEqual(len(vc[0:500]), 250) #should there me an error or warning message instead?


# a valid vedacollection ID
VC_ID = 'LC80370302014268LGN00'
# a valid datapoint from the above
DP_ID = 'LC80370302014268LGN00'

def force(r1, r2):
    return True

my_vcr = vcr.VCR()
my_vcr.register_matcher('force', force)
my_vcr.match_on = ['force']

# How to use the mock_gbdx_session and vcr to create unit tests:
# 1. Add a new test that is dependent upon actually hitting GBDX APIs.
# 2. Decorate the test with @vcr appropriately
# 3. Replace "dummytoken" with a real gbdx token
# 4. Run the tests (existing test shouldn't be affected by use of a real token).  This will record a "cassette".
# 5. Replace the real gbdx token with "dummytoken" again
# 6. Edit the cassette to remove any possibly sensitive information (s3 creds for example)

@skip("skipping for now")
class VedaCollectionTest_vcr(unittest.TestCase):
    @skip
    def test_vedacollection_new(self):
        vc = VedaCollection('name')
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertEqual(vc.imshape, (0, 256, 256))
        self.assertEqual(vc.mlType, 'classification')
        self.assertEqual(vc.bounds, None)
        self.assertEqual(vc.count, 0)
        
    @skip
    @my_vcr.use_cassette('tests/unit/cassettes/test_vc_search.yaml', filter_headers=['authorization'])
    def test_vc_search(self):
        results = search()
        vc = VedaCollection.from_doc(results[0])
        self.assertTrue(isinstance(vc, VedaCollection))
        self.assertGreater(vc.count, 0)
        self.assertEqual(type(shape(vc)), 'Polygon')

    @skip
    @my_vcr.use_cassette('tests/unit/cassettes/test_vedacollection_id.yaml', filter_headers=['authorization'])
    def test_vedacollection_id(self):
        vc = VedaCollection.from_id(VC_ID)
        self.assertEqual(vc.id, vc_id)
        self.assertEqual(vc.imshape, (3, 128, 128))
        self.assertEqual(vc.mlType,'classification')


@skip("skipping for now")
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
        dps = vc.fetch_points(5)
        self.assertEqual(len(dps), 5)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_points_index.yaml', filter_headers=['authorization'])
    def test_fetch_points_offset(self):
        dps = vc.fetch_points(10,5)
        self.assertEqual(len(dps), 10)


    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_ids.yaml', filter_headers=['authorization'])
    def test_fetch_ids(self):
        ids = vc.fetch_ids(page_size=50)
        self.assertEqual(len(ids), 50)
