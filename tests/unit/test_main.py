''' Tests for functions in main.py '''

import os, sys
from auth_mock import conn, my_vcr

from pyveda.veda.api import VedaCollectionProxy
from pyveda.vedaset import VedaStream
from pyveda.vedaset import VedaBase

from unittest.mock import patch

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

    @patch('pyveda.main.from_geo')
    def test_createfromgeojson(self, from_geo):
        ''' Test the create_from_geojson method '''
        # TODO: create fixtures for this kind of thing?
        mock_doc = {
            "name":              'mockname',
            "classes":          [],
            "dataset_id":       '234',
            "dtype":            np.dtype('uint8'),
            "userId":           'foo',
            "imshape":          [],
            "releases":         'releases',
            "mltype":           'classification',
            "tilesize":         [],
            "image_refs":       [],
            "sensors":          [],
            "bounds":           [],
            "public":           True,
            "count":            3,
            "percent_cached":   1,
            "background_ratio": 1.0
        }
        from_geo.return_value = {'properties': mock_doc}
        geojson = {}
        image = []
        vcp = pv.create_from_geojson(geojson, image, mock_doc['name'])
        self.assertTrue(isinstance(vcp, VedaCollectionProxy))
