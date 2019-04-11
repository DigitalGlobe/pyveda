''' Tests for Veda data accessor handlers '''

from pyveda.fetch.handlers import *
import unittest
from unittest import skip


class VedaBaseLabelTest(unittest.TestCase):

    def test_classification(self):
        from sampledata import classification_item
        label = BinaryClassificationHandler._payload_handler(classification_item, klasses=['house','car'])
        self.assertEqual(label, [1,1])

    def test_segmentation(self):
        from sampledata import segmentation_item
        label = InstanceSegmentationHandler._payload_handler(segmentation_item, klasses=['building'], out_shape = [256,256])
        # Point inside is classified
        self.assertEqual(label[1][80], 0.0)
        # Point on vertex is classified
        self.assertEqual(label[52][84], 0.0)
        # Point outside is background value
        self.assertEqual(label[255][255], 0.0)


    def test_object_detection(self):
        from sampledata import objd_item
        label = ObjectDetectionHandler._payload_handler(objd_item, klasses=['building', 'damaged building'], out_shape = [256,256])
        self.assertEqual(label[1], [])
        self.assertEqual(label[0][0], [235,62,256,117])
