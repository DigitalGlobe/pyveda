'''
Tests for training objects - VedaCollection & DataPoint
'''

# path hack for veda installs
import os, sys
sys.path.append("..")

from pyveda import VedaCollection, DataPoint, search
from shapely.geometry import shape
import vcr
import unittest


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


class DataPointFetchTest(unittest.TestCase):

    #@my_vcr.use_cassette('tests/unit/cassettes/test_datapoint_fetch.yaml', filter_headers=['authorization'])
    def setUp(self):
        vc = VedaCollection.from_id(VC_ID) 
        self.dp = vc.fetch(DP_ID)

    def test_datapoint(self):
        self.assertTrue(isinstance(self.dp, DataPoint))
        assert self.dp.id == DP_ID 
        assert type(shape(self.dp)) == 'Polygon'
        assert self.dp.data['dataset_id'] == VC_ID
