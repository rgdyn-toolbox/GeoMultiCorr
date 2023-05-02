import sys
sys.path.append("..")
import unittest

from src.GeoMultiCorr.gmc_session import GMC_Session

TEST_SESSION_PATH = '/media/duvanelt/TD002/sandbox_gmc/vanoise'

class Test_GMC_Session(unittest.TestCase):
    
    def setUp(self):
        self.test_session = GMC_Session(TEST_SESSION_PATH)

    def test_is_session(self):
        self.assertIsInstance(self.test_session, GMC_Session)

if __name__ == '__main__':
    unittest.main()