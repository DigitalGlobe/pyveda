''' Tests for VedaBase '''

import os, sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

from pyveda.vedaset import VedaBase
import unittest

class VedaBaseTest(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.id = '67a16de1-7baf-44bf-a779-2bf97a37c3bd'
        self.h5 = './test.h5'
    
    @classmethod
    def tearDownClass(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    @my_vcr.use_cassette('tests/unit/cassettes/test_vedabase.yaml', filter_headers=['authorization'])
    def test_vedabase(self):
        vc = pv.from_id(self.id)
        vb = pv.store(self.h5, dataset_id=self.id, count=10)
        self.assertTrue(isinstance(vb, VedaBase))
        self.assertEqual(len(vb.train), 7)
        #self.assertEqual(len(list(vb.train)), 7)
        self.assertEqual(len(list(vb.train[:])), 7)
        self.assertEqual(vb.image_shape, vc.imshape)
        self.assertEqual(vb.mltype, vc.mltype)
