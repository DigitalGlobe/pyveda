''' Tests for funcitions in main.py '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

import pyveda as pv
from pyveda.veda.api import VedaCollectionProxy
from pyveda.vedaset import VedaStream
from pyveda.vedaset import VedaBase

import unittest

#from urllib.error import HTTPError
from requests.exceptions import HTTPError

class MainFunctionsTest(unittest.TestCase):
    def setUp(self):
        id = 'a15f07fa-85d4-4af7-a22e-7e0c50c7dfb9'
        self.vcp = VedaCollectionProxy.from_id(id)
        self.h5 = './test.h5'
        self.vs = vb.load(self.vcp)
        
    def tearDown(self):
            try:
                os.remove(self.h5)
            except OSError:
                pass

    def test_mapcontainssubmap(self):
        pass

    def test_search(self):
        for x in pv.search():
            self.assertTrue(isinstance(x, VedaCollectionProxy))

    def test_datasetexists(self):
        self.assertTrue(isinstance(pv.dataset_exists('a15f07fa-85d4-4af7-a22e-7e0c50c7dfb9'), VedaCollectionProxy))
        self.assertTrue(isinstance(pv.dataset_exists(dataset_name = 'Austin Segmentation'), VedaCollectionProxy))
        #self.assertRaises(HTTPError, pv.dataset_exists, 'a')
        #self.assertRaises(HTTPError, pv.dataset_exists, None, dataset_name = 'xyz')

    def test_load(self):
        self.assertRaises(ValueError, pv.store, self.vcp)
        self.assertRaises(HTTPError, pv.store, dataset_id = id, filename = self.h5)
        self.assertTrue(isinstance(pv.load('ce6c0c66-f679-4bb5-aa14-eaa1579fe282'), VedaStream))

    def test_store(self):
        self.assertRaises(ValueError, pv.store, self.vcp)
        self.assertRaises(HTTPError, pv.store, dataset_id = id, filename = self.h5)
        #self.assertTrue(isinstance(pv.store(dataset_id = 'a15f07fa-85d4-4af7-a22e-7e0c50c7dfb9', filename = self.h5, count = 10), VedaBase))

    def test_loadexisting(self):
        self.assertTrue(isinstance(pv.load(self.vcp), VedaStream))

    def test_loadstore(self):
        self.assertTrue(isinstance(pv.store(dataset_id = 'a15f07fa-85d4-4af7-a22e-7e0c50c7dfb9', filename = self.h5, count = 10), VedaBase))

    def test_createfromgeojson(self):
        pass
