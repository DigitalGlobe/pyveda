'''
Tests for DataPoint that don't rely on the server
'''
import os, sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

from pyveda.models.main import PredictionSet, PredictionSampleClient
import json
import unittest

class PredictionTest(unittest.TestCase):
    @classmethod 
    def setUpClass(self):
        self._id = '09a64c45-b6d3-4620-ad65-0512eae9f261'

    @my_vcr.use_cassette('tests/unit/cassettes/test_prediction_set.yaml', filter_headers=['authorization'])
    def test_prediction_set(self):
        pass
        # TODO cant pass until deployed to dev
        #pset = PredictionSet.from_id(self._id)
        #self.assertIsInstance(pset, PredictionSet)
        ## public properties
        #self.assertEqual(pset.bounds, [-97.7651149214, 30.2704546428, -97.7620965959, 30.2734729683])
        #self.assertEqual(pset.classes, ['building'])
        #self.assertEqual(pset.mltype, 'segmentation')

    

    
