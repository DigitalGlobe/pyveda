'''
Tests for VedaBase that don't rely on the server
'''

# path hack for veda installs
import os, sys
sys.path.append("..")
os.environ["SANDMAN_API"] = "https://veda-api-development.geobigdata.io"

from pyveda import DataPoint, VedaCollection
import json
from shapely.geometry import shape, box
from shapely.geometry.polygon import Polygon
import unittest

class VedaBaseTest(unittest.TestCase):
    
