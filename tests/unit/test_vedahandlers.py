''' Tests for Veda data accessor handlers '''

from pyveda.fetch.handlers import ClassificationHandler, SegmentationHandler, ObjDetectionHandler
import unittest
from unittest import skip


class VedaBaseLabelTest(unittest.TestCase):

    def test_classification(self):
        from sampledata import classification_item
        label = ClassificationHandler._payload_handler(classification_item, klasses=['house','car'])
        self.assertEqual(label, [1,1])

    def test_segmentation(self):
        from sampledata import segmentation_item
        label = SegmentationHandler._payload_handler(segmentation_item, klasses=['building'], out_shape = [256,256])
        # Point inside is classified
        self.assertIs(label[1][80], 1)
        # Point on vertex is classified
        self.assertIs(label[52][84], 1)
        # Point outside is background value
        self.assertIs(label[256][256], 0)


    def test_object_detection(self):
        from sampledata import objd_item
        label = ObjDetectionHandler._payload_handler(objd_item, klasses=['building', 'damaged building'], out_shape = [256,256])
        self.assertEqual(label[1], [])
        self.assertEqual(label[0][0], [235,62,256,117])