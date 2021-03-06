''' Tests for VedaBase '''

import os, sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

from pyveda.vedaset import VedaBase
from pyveda.io.io import build_vedabase
from pyveda.vedaset.store.vedabase import H5SampleArray
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported

import unittest
from unittest import skip

@skip
class VCtoVBTest(unittest.TestCase):

    # this won't record the async fetches!
    @my_vcr.use_cassette('tests/unit/cassettes/test_vedabase_setup.yaml', filter_headers=['authorization'])
    def setUp(self):
        vc_id = '82553d2f-9c9c-46f0-ad9f-1a27a8673637'
        self.h5 = './test.h5'
        try:
            os.remove(self.h5)
        except OSError:
            pass
        self.vc = VedaCollection.from_id(vc_id)
        self.vb = self.vc.store(self.h5, size=10)

    def tearDown(self):
        self.vb.close()

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

    def test_VC_to_VB(self):
        self.assertTrue(isinstance(self.vb, VedaBase))
        # VBs support __len__ but maybe we should test these cases too
        #self.assertEqual(len(list(self.vb.train)), 7)
        #self.assertEqual(len(list(self.vb.train[:])), 7)
        self.assertEqual(len(self.vb.train), 7)
        self.assertEqual(self.vb.image_shape, self.vc.imshape)
        self.assertEqual(self.vb.mltype, self.vc.mltype)

class VedaBaseTest(unittest.TestCase):

    def setUp(self):
        self.h5 = './base.h5'
        try:
            os.remove(self.h5)
        except OSError:
            pass

    def tearDown(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    @my_vcr.use_cassette('tests/unit/cassettes/test_vedabase.yaml', filter_headers=['authorization'])
    def test_vb_init(self):
        coll = pv.from_id('67a16de1-7baf-44bf-a779-2bf97a37c3bd')
        count = 10
        partition=[70,20,10]
        vb = VedaBase.from_path(self.h5,
                          mltype=coll.mltype,
                          classes=coll.classes,
                          image_shape=coll.imshape,
                          image_dtype=coll.dtype)

        self.assertEqual(type(vb), VedaBase)
        self.assertEqual(vb.mltype.name, coll.mltype)
        self.assertEqual(vb.classes, coll.classes)
        self.assertEqual(vb.image_shape, coll.imshape)
        self.assertEqual(vb.image_dtype, coll.dtype)
        #with self.assertRaises(FrameworkNotSupported):
        #    vb.framework = 'foo'
        #self.assertEqual(vb.framework, self.framework)
        #vb.framework = 'Keras'
        #self.assertEqual(vb.framework, 'Keras')
        self.assertEqual(type(vb.train), H5SampleArray)
        self.assertEqual(type(vb.test), H5SampleArray)
        self.assertEqual(type(vb.validate), H5SampleArray)
