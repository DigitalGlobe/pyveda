''' Tests for funcitions in main.py '''

import os, sys
from auth_mock import conn, my_vcr
from gbdxtools import CatalogImage

from pyveda.veda.api import VedaCollectionProxy
from pyveda.vedaset import VedaStream
from pyveda.vedaset import VedaBase

import numpy as np
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

import unittest
from requests.exceptions import HTTPError

class MainFunctionsTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.id = '1245073f-dae9-40ac-ac5c-52ca349ae8dd'
        #self.vcp = VedaCollectionProxy.from_id(id)
        self.h5 = './test.h5'

    @classmethod
    def tearDownClass(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    def test_mapcontainssubmap(self):
        pass

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_search.yaml', filter_headers=['authorization'])
    def test_search(self):
        for x in pv.search():
            self.assertTrue(isinstance(x, VedaCollectionProxy))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_from_name.yaml', filter_headers=['authorization'])
    def test_from_name(self):
        self.assertTrue(isinstance(pv.from_name('Austin Segmentation'), VedaCollectionProxy))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_from_id.yaml', filter_headers=['authorization'])
    def test_from_id(self):
        vcp = pv.from_id(self.id)
        self.assertTrue(isinstance(pv.from_id(self.id), VedaCollectionProxy))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_open.yaml', filter_headers=['authorization'])
    def test_open(self):
        vcp = pv.from_id(self.id)
        #self.assertRaises(ValueError, pv.store, vcp)
        #self.assertRaises(HTTPError, pv.store, dataset_id = self.id, filename = self.h5)
        self.assertTrue(isinstance(pv.open(self.id), VedaStream))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_store.yaml', filter_headers=['authorization'])
    def test_store(self):
        vcp = pv.from_id(self.id)
        self.assertRaises(ValueError, pv.store, vcp)
        #self.assertRaises(HTTPError, pv.store, dataset_id = self.id, filename = self.h5)
        self.assertTrue(isinstance(pv.store(dataset_id = self.id, filename = self.h5, count = 10), VedaBase))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_loadexisting.yaml', filter_headers=['authorization'])
    def test_loadexisting(self):
        self.assertTrue(isinstance(pv.open(self.id), VedaStream))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_loadstore.yaml', filter_headers=['authorization'])
    def test_loadstore(self):
        store = pv.store(dataset_id = self.id, filename = self.h5, count = 10)
        self.assertTrue(isinstance(store, VedaBase))

    @my_vcr.use_cassette('tests/unit/cassettes/test_main_createfromgeojson.yaml', filter_headers=['authorization'])
    def test_createfromgeojson(self):
        geojson = {'features': [{'type': 'Feature', 'properties': {'Name': None, 'label': 'american_football_field'}, 'geometry': {'type': 'Polygon', 'coordinates': [[[-122.4907898268108, 37.778191163762486], [-122.49129463633359, 37.7781838052681], [-122.49133721678479, 37.77898525977172], [-122.49083240181278, 37.778992618408225], [-122.4907898268108, 37.778191163762486]]]}}, {'type': 'Feature', 'properties': {'Name': None, 'label': 'american_football_field'}, 'geometry': {'type': 'Polygon', 'coordinates': [[[-122.1102008575904, 37.8440139386442], [-122.1113937618264, 37.843225245649634], [-122.11200867572458, 37.84375777445076], [-122.11075646129174, 37.84455775662023], [-122.1102008575904, 37.8440139386442]]]}}]}
        # TODO: do we really want all of CatalogImage here?
        image = CatalogImage('103001002300F900')
        name = 'name2'
        background_ratio = 2.0
        pv.config.set_dev()
        vcp = pv.create_from_geojson(geojson, image, name, background_ratio=background_ratio)
        self.assertEqual(vcp.background_ratio, background_ratio) # # pylint: disable=no-member
