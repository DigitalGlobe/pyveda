''' Tests for funcitions in main.py '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

import pyveda as pv
from pyveda.veda.api import VedaCollectionProxy
from pyveda.vedaset import VedaStream
from pyveda.vedaset import VedaBase

import unittest

from requests.exceptions import HTTPError

class MainFunctionsTest(unittest.TestCase):
    def setUp(self):
        id = 'f9548932-29d5-4668-bf9b-3644b335491b'
        self.vcp = VedaCollectionProxy.from_id(id)
        self.h5 = './test.h5'

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

    def test_from_name(self):
        self.assertTrue(isinstance(pv.from_name('Austin Segmentation'), VedaCollectionProxy))

    def test_from_id(self):
        self.assertTrue(isinstance(pv.from_id('f9548932-29d5-4668-bf9b-3644b335491b'), VedaCollectionProxy))

    def test_load(self):
        self.assertRaises(ValueError, pv.store, self.vcp)
        self.assertRaises(HTTPError, pv.store, dataset_id = id, filename = self.h5)
        self.assertTrue(isinstance(pv.open('f9548932-29d5-4668-bf9b-3644b335491b'), VedaStream))

    def test_store(self):
        self.assertRaises(ValueError, pv.store, self.vcp)
        self.assertRaises(HTTPError, pv.store, dataset_id = id, filename = self.h5)
        self.assertTrue(isinstance(pv.store(dataset_id = 'f9548932-29d5-4668-bf9b-3644b335491b', filename = self.h5, count = 10), VedaBase))

    def test_loadexisting(self):
        self.assertTrue(isinstance(pv.open('f9548932-29d5-4668-bf9b-3644b335491b'), VedaStream))

    def test_loadstore(self):
        self.assertTrue(isinstance(pv.store(dataset_id = 'f9548932-29d5-4668-bf9b-3644b335491b', filename = self.h5, count = 10), VedaBase))

    def test_createfromgeojson(self):
        pass
