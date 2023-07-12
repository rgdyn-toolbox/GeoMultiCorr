import unittest
import geopandas as gpd

import src.geomulticorr.session
import src.geomulticorr.spine

test_create_protomap = False

class TestSession(unittest.TestCase):

    def setUp(self):
        self.path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
        self.test_session = src.geomulticorr.session.Session(self.path_to_test_session)
        self.test_pzone_name = 'iseran'
        self.test_spine_id = 'ISFR730486'

    def test_is_session(self):
        self.assertIsInstance(self.test_session, src.geomulticorr.session.Session)

    def test_get_thumbs(self):
        self.assertIsInstance(self.test_session.get_thumbs(), list)
        self.assertIsInstance(self.test_session.get_thumbs(self.test_pzone_name), list)

    def test_get_pairs(self):
        self.assertIsInstance(self.test_session.get_pairs(), list)
        self.assertIsInstance(self.test_session.get_pairs(self.test_pzone_name), list)

    def test_get_pzones(self):
        self.assertIsInstance(self.test_session.get_pzones(), list)
        self.assertIsInstance(self.test_session.get_pzones(self.test_pzone_name), list)

    def test_get_geomorphs(self):
        self.assertIsInstance(self.test_session.get_geomorphs(), list)
        self.assertIsInstance(self.test_session.get_geomorphs(self.test_pzone_name), list)

    def test_get_spine(self):
        self.assertIsInstance(self.test_session.get_spine(self.test_spine_id), src.geomulticorr.spine.Spine)

    def test_get_protomap(self):
        self.assertIsInstance(self.test_session.get_protomap(), gpd.GeoDataFrame)

    def test_update_pairs(self):
        self.assertIsInstance(self.test_session.update_pairs(), gpd.GeoDataFrame)
        self.assertIsInstance(self.test_session.update_pairs(), gpd.GeoDataFrame)

    def test_update_thumbs(self):
        self.assertIsInstance(self.test_session.update_thumbs(), gpd.GeoDataFrame)
        self.assertIsInstance(self.test_session.update_thumbs(), gpd.GeoDataFrame)

    def test_create_new_protomap(self):
        if test_create_protomap:
            self.assertIsInstance(self.test_session.create_new_protomap("/media/duvanelt/RMS_IMG/rgdyn-france-images/AERIAL_REPOSITORY", save_in_geodatabase=False), gpd.GeoDataFrame)

    def test_copy_geodatabase(self):
        self.assertEqual(self.test_session.copy_geodb(), True)

    """
    test_session.sieve() is not tested
    test_session.pr_full() is not tested
    test_session.update_vector_data() is not tested
    """

if __name__ == '__main__':
    unittest.main()