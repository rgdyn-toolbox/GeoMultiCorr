import unittest

import pandas as pd

from osgeo import gdal
import telenvi

import src.geomulticorr.thumb

test_create_protomap = False

class TestThumb(unittest.TestCase):

    def setUp(self):
        self.path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
        self.test_session = src.geomulticorr.session.Session(self.path_to_test_session)
        self.test_thumb = self.test_session.get_thumbs()[0]

    def test_is_thumb(self):
        self.assertIsInstance(self.test_thumb, src.geomulticorr.thumb.Thumb)

    def test_to_pdserie(self):
        self.assertIsInstance(self.test_thumb.to_pdserie(), pd.Series)

    def test_to_geoim(self):
        self.assertIsInstance(self.test_thumb.get_ds(), gdal.Dataset)

    def test_to_geoim(self):
        self.assertIsInstance(self.test_thumb.get_geoim(), telenvi.GeoIm.GeoIm)

    def test_show(self):
        self.test_thumb.show()

if __name__ == '__main__':
    unittest.main()