import unittest
import geopandas as gpd

import src.geomulticorr.session

test_create_protomap = False

class TestPzone(unittest.TestCase):

    def setUp(self):
        self.path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
        self.test_session = src.geomulticorr.session.Session(self.path_to_test_session)
        self.test_pzone = self.test_session.get_pzones('ribon')[0]

    def test_is_pzone(self):
        self.assertIsInstance(self.test_pzone, src.geomulticorr.pzone.Pzone)

    def test_get_thumbs(self):
        self.assertIsInstance(self.test_pzone.get_thumbs(), list)
        self.assertIsInstance(self.test_pzone.get_thumbs(2018), list)

    def test_get_pairs(self):
        self.assertIsInstance(self.test_pzone.get_pairs(), list)
    
    def test_get_valid_thumbs(self):
        self.assertIsInstance(self.test_pzone.get_valid_thumbs(), list)
    
    def test_get_valid_pairs(self):
        self.assertIsInstance(self.test_pzone.get_valid_pairs(), list)
    
    # def test_pz_full(self):
    #     self.assertIsInstance(self.test_pzone.pz_full(), dict)
