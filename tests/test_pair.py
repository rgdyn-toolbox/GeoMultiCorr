import unittest
from pathlib import Path

import pandas as pd

import geopandas as gpd
from osgeo import gdal
import telenvi

import src.geomulticorr.thumb

test_create_protomap = False

class TestPair(unittest.TestCase):

    def setUp(self):
        self.path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
        self.test_session = src.geomulticorr.session.Session(self.path_to_test_session)
        self.test_pair = self.test_session.get_pairs(['lou', 2014, 2018])[0]

    def test_is_pair(self):
        self.assertIsInstance(self.test_pair, src.geomulticorr.pair.Pair)

    def test_get_status(self):
        self.assertIsInstance(self.test_pair.get_status(), str)

    def test_to_pdserie(self):
        self.assertIsInstance(self.test_pair.to_pdserie(), pd.Series)

    def test_clip(self):
        self.assertEqual(self.test_pair.clip(), True)

    def test_corr(self):
        self.assertEqual(self.test_pair.corr(), True)

    def test_save_corrdata(self):
        self.assertEqual(self.test_pair.save_corrdata(), True)

    def test_compute_magnitude(self):
        self.assertIsInstance(self.test_pair.compute_magnitude(), telenvi.GeoIm.GeoIm)    

    def test_vectorize(self):
        self.assertIsInstance(self.test_pair.vectorize(output_pixel_size=50), gpd.GeoDataFrame)

    def test_pa_full(self):
        self.assertEqual(self.test_pair.pa_full(), True)

    def test_get_interesting_geoim(self):
        self.assertIsInstance(self.test_pair.get_interesting_geoim('vm'), telenvi.GeoIm.GeoIm)

    """
    not tested :
        get_slice
        get_slices
    """

if __name__ == '__main__':
    unittest.main()