import unittest
import geomulticorr.src.session

path_to_test_session = '/media/duvanelt/TD002/sandbox_gmc/vanoise'

class TestSession(unittest.TestCase):

    def setUp(self):
        self.valid_test_session = geomulticorr.src.session.Session(path_to_test_session)

    def test_is_session(self):
        self.assertIsInstance(self.valid_test_session, geomulticorr.src.session.Session)

if __name__ == '__main__':
    unittest.main()



















































"""

"""
# %%
