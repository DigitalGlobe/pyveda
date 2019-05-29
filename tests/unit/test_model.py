'''
Tests for DataPoint that don't rely on the server
'''
import os, sys
from auth_mock import conn, my_vcr
import pyveda as pv
pv.config.set_dev()
pv.config.set_conn(conn)

from pyveda.models import Model
import json
import unittest

class ModelTest(unittest.TestCase):
    @classmethod 
    def setUpClass(self):
        self.vc_id = '1245073f-dae9-40ac-ac5c-52ca349ae8dd'
        self.model_id = 'eb1e1e0b-c48e-494f-b56a-4bb0734c98fc'

    @my_vcr.use_cassette('tests/unit/cassettes/test_model_create.yaml', filter_headers=['authorization'])
    def test_model_create(self):
        vc = pv.from_id(self.vc_id)
        archive = 'dummy.tar.gz'
        model = Model('CHELM SEG PREDICTIONS', 
            archive=archive,
            library="keras",               
            training_set=vc,
            imshape=(256,256,3),
            mltype='segmentation',
            channels_last=True
        )  
        self.assertIsInstance(model, Model)
        # public properties
        self.assertTrue(model.channels_last)
        self.assertEqual(model.library, "keras")
        self.assertEqual(vc.bounds, model.bounds)
        self.assertEqual(model.imshape, (256,256,3))
        self.assertEqual(vc.mltype, model.mltype)

    

    
