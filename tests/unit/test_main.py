''' Tests for funcitions in Main.py '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

import pyveda as pv
from pyveda.veda.api import VedaCollectionProxy


class MainFunctionsTest(unittest.TestCase):
    def setUp(self):

    #def tearDown(self):

    def test_search(self):
        for x in pv.search():
            self.assertisInstance(x, VedaCollectionProxy)

    def test_datasetexists(self):
        self.assertisInstance(pv.dataset_exists('a15f07fa-85d4-4af7-a22e-7e0c50c7dfb9'), VedaCollectionProxy)
        self.assertisInstance(pv.dataset_exists(dataset_name = 'Austin Segmentation'), VedaCollectionProxy)
        self.assertRaises(HTTPError, pv.dataset_exists, 'a')
        self.assertRaises(HTTPError, pv.dataset_exists, None, dataset_name = 'xyz')

    def test_load(self):

    def test_store(self):

    def test_loadexisting(self):

    def test_loadstore(self):

    def test_createfromgeojson(self):
