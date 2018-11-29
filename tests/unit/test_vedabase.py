''' Tests for VedaBase '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import VedaCollection
from pyveda.vedaset import VedaBase
from pyveda.vedaset.store.vedabase import WrappedDataNode
from pyveda.exceptions import LabelNotSupported, FrameworkNotSupported
from auth_mock import gbdx
import vcr
import unittest
from unittest import skip

def force(r1, r2):
    return True

my_vcr = vcr.VCR()
my_vcr.register_matcher('force', force)
my_vcr.match_on = ['force']


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
        try:
            os.remove(self.h5)
        except OSError:
            pass

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
        self.mltype = 'classification'
        self.klasses = ['a', 'b']
        self.image_shape = [3, 10, 10]
        self.image_dtype = "uint8"
        self.framework = "PyTorch"

    def tearDown(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    def test_vb_init(self):
        with VedaBase(self.h5,
                        mltype=self.mltype,
                        klasses=self.klasses,
                        image_shape=self.image_shape, 
                        image_dtype=self.image_dtype,
                        framework=self.framework) as vb:
            self.assertEqual(type(vb), VedaBase)
            self.assertEqual(vb.mltype, self.mltype)
            self.assertEqual(vb.classes, self.klasses)
            self.assertEqual(vb.image_shape, self.image_shape)
            self.assertEqual(vb.image_dtype, self.image_dtype)
            with self.assertRaises(FrameworkNotSupported):
                vb.framework = 'foo'
            self.assertEqual(vb.framework, self.framework)
            vb.framework = 'Keras'
            self.assertEqual(vb.framework, 'Keras')
            self.assertEqual(len(vb), 0) 
            self.assertEqual(type(vb.train), WrappedDataNode)
            self.assertEqual(type(vb.test), WrappedDataNode)
            self.assertEqual(type(vb.validate), WrappedDataNode)