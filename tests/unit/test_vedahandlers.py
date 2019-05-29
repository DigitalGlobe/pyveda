''' Tests for Veda data accessor handlers '''

from pyveda.io.remote.handlers import *
from pyveda.vedaset.vedaset import VedaSet
import unittest
from unittest import skip


class VedaBaseLabelTest(unittest.TestCase):

    def test_classification(self):
        from sampledata import classification_item
        vset = VedaSet()
        vset.classes = ['house', 'car']
        label = BinaryClassificationHandler(vset)._payload_handler(classification_item)
        self.assertEqual(label, [1,1])

    def test_segmentation(self):
        from sampledata import segmentation_item
        vset = VedaSet()
        vset.classes = ['building']
        vset.image_shape = [256, 256]
        label = InstanceSegmentationHandler(vset)._payload_handler(segmentation_item)
        # Point inside is classified
        self.assertEqual(label[1][80], 0.0)
        # Point on vertex is classified
        self.assertEqual(label[52][84], 0.0)
        # Point outside is background value
        self.assertEqual(label[255][255], 0.0)


    def test_object_detection(self):
        from sampledata import objd_item
        vset = VedaSet()
        vset.classes = ['building', 'damaged building']
        vset.image_shape = [256, 256]
        label = ObjectDetectionHandler(vset)._payload_handler(objd_item)
        self.assertEqual(label[1], [])
        self.assertEqual(label[0][0], [235,62,256,117])
