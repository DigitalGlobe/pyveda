''' Tests for VedaBase '''

import os, sys
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import VedaCollection
from pyveda.db import VedaBase
import json
from shapely.geometry import shape, box
from shapely.geometry.polygon import Polygon
import unittest

class VedaBaseTest(unittest.TestCase):
    def setUp(self):
        vc_id = '82553d2f-9c9c-46f0-ad9f-1a27a8673637'
        dp_id = 'c5942231-dd6d-4ab8-9fce-04d28aa560d8'
        self.h5 = './test.h5'
        self.vc = VedaCollection.from_id(vc_id)
        self.vb = self.vc.store(self.h5, size=10)

    def tearDown(self):
        try:
            os.remove(self.h5)
        except OSError:
            pass

    def test_vedabase(self):
        self.assertTrue(isinstance(self.vb, VedaBase))
        self.assertEqual(len(self.vb.train), 7)
        #self.assertEqual(len(list(self.vb.train)), 7)
        self.assertEqual(len(list(self.vb.train[:])), 7)
        self.assertEqual(self.vb.image_shape, self.vc.imshape)
        self.assertEqual(self.vb.mltype, self.vc.mltype)