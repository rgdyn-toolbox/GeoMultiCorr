import unittest
from pathlib import Path

import src.geomulticorr.session

class TestGeomorph(unittest.TestCase):

    def setUp(self):
        self.path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
        self.test_session = src.geomulticorr.session.Session(self.path_to_test_session)
        self.test_pzone_name = 'laurichard'
        self.test_geomorph = self.test_session.get_geomorphs(self.test_pzone_name)[0]
        self.save_figure_path = Path('/home/duvanelt/test_graphe.png')

    def test_is_geomorph(self):
        self.assertIsInstance(self.test_session, src.geomulticorr.session.Session)

    def test_get_thumbs(self):
        self.assertIsInstance(self.test_geomorph.get_thumbs(), list)
        self.assertIsInstance(self.test_geomorph.get_thumbs(self.test_pzone_name), list)

    def test_get_pairs(self):
        self.assertIsInstance(self.test_geomorph.get_pairs(), list)

    def test_get_spines(self):
        self.assertIsInstance(self.test_geomorph.get_spines(), list)

    def test_get_pairs_complete(self):
        self.assertIsInstance(self.test_geomorph.get_pairs_complete(), list)

    def test_get_mean_disp_on_pair(self):
        test_pair = self.test_geomorph.get_pairs_complete()[0]
        self.assertIsInstance(self.test_geomorph.get_mean_disp_on_pair(test_pair.pa_magn_path), float)

    def test_show_mean_velocities(self):
        self.test_geomorph.show_mean_velocities(str(self.save_figure_path))
        assert self.save_figure_path.exists(), 'graphe non généré'
        self.save_figure_path.unlink()
        
if __name__ == '__main__':
    unittest.main()