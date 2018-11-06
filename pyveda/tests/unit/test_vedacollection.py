'''
Tests for training objects - VedaCollection & DataPoint
'''

from pyveda import VedaCollection, DataPoint, search
from shapely.geometry import shape
import vcr
import unittest
import os 

os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

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


class VedaCollectionTest(unittest.TestCase):

    def test_vedacollection_new(self):
        vc = VedaCollection('name')
        self.assertTrue(isinstance(vc, VedaCollection))
        assert vc.imshape == (0, 256, 256)
        assert vc.mlType == 'classification'
        assert vc.bounds == None
        assert vc.count == 0

    @my_vcr.use_cassette('tests/unit/cassettes/test_vc_search.yaml', filter_headers=['authorization'])
    def test_vc_search(self):
        results = search()
        vc = VedaCollection.from_doc(results[0]) 
        self.assertTrue(isinstance(vc, VedaCollection))
        assert vc.count > 0
        assert type(shape(vc)) == 'Polygon'

    @my_vcr.use_cassette('tests/unit/cassettes/test_vedacollection_id.yaml', filter_headers=['authorization'])
    def test_vedacollection_id(self):
        vc = VedaCollection.from_id(VC_ID) 
        assert vc.id == vc_id 
        assert vc.imshape == (3, 128, 128)
        assert vc.mlType == 'classification'

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
        assert len(dps = 5)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_points_index.yaml', filter_headers=['authorization'])
    def test_fetch_points_offset(self):
        dps = vc.fetch_points(10,5)
        assert len(dps = 10)

    @my_vcr.use_cassette('tests/unit/cassettes/test_vcfetch_fetch_ids.yaml', filter_headers=['authorization'])
    def test_fetch_ids(self):
        ids = vc.fetch_ids(page_size=50)
        assert len(ids) == 50 

