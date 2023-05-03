import unittest
import geomulticorr.src.session

path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'
pzone_name = 'iseran'

class TestSession(unittest.TestCase):

    def setUp(self):
        self.test_session = geomulticorr.src.session.Session(path_to_test_session)

    def test_is_session(self):
        self.assertIsInstance(self.test_session, geomulticorr.src.session.Session)

    def test_get_thumbs(self):
        self.assertIsInstance(self.test_session.get_thumbs(), list)
        self.assertIsInstance(self.test_session.get_thumbs(pzone_name), list)

    def test_get_pairs(self):
        self.assertIsInstance(self.test_session.get_pairs(), list)
        self.assertIsInstance(self.test_session.get_pairs(pzone_name), list)

if __name__ == '__main__':
    unittest.main()



















































"""

"""
# %%
