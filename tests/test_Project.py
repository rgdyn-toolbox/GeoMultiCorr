import sys
import os
from pathlib import Path
sys.path.append("..")
from src.gmc_project import GMC_Project
import json
import unittest

D = json.load(open(Path(__file__).with_name('data_for_testing.json')))

class TestProject(unittest.TestCase):

    def setUp(self):
        self.to_load = GMC_Project(D['to_load'])
        self.to_create = GMC_Project(D['to_create'])
        try:
            self.to_avoid = GMC_Project(D['to_avoid'])
        except ValueError:
            self.to_avoid = None
        try:
            self.invalid = GMC_Project(D['invalid'])
        except AssertionError:
            self.invalid = None

    def test_init(self):
        self.assertIsInstance(self.to_load,   GMC_Project)
        self.assertIsInstance(self.to_create, GMC_Project)
        self.assertNotIsInstance(self.to_avoid, GMC_Project)
        self.assertNotIsInstance(self.invalid, GMC_Project)

if __name__ == '__main__':
    unittest.main()