import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Scripts.NSBasic import *
import unittest
from Scripts.NSBasic import NSBasic, NS
from Modules.NNet import NeuralNetwork as NNet

class TestNSBasic(unittest.TestCase):
    def setUp(self) -> None:
        self.NSB = NSBasic()
    
    def test_init(self):
        self.assertEqual(self.NSB.set_P, {})
        self.assertEqual(self.NSB.set_N, {})
        self.assertEqual(self.NSB.set_Z, {})
        self.assertEqual(self.NSB.set_U, {})
        
    def test_set_P(self):
        self.NSB.set_P = {1: 1}
        self.assertEqual(self.NSB.set_P, {1: 1})
        
    def test_set_N(self):
        self.NSB.set_N = {1: 1}
        self.assertEqual(self.NSB.set_N, {1: 1})
        
    def test_set_Z(self):
        self.NSB.set_Z = {1: 1}
        self.assertEqual(self.NSB.set_Z, {1: 1})
        
    def test_set_U(self):
        self.NSB.set_U = {1: 1}
        self.assertEqual(self.NSB.set_U, {1: 1})
        
    def test_network(self):
        network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.NSB.get_network(network)
        self.assertEqual(self.NSB.network, network)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, -2).display())
        self.assertEqual(self.NSB.set_U[1][31].display(), NeuronStatus(1, 31, -2).display())
        self.assertEqual(self.NSB.set_U[2][0].display(), NeuronStatus(2, 0, -2).display())
                                              
class TestNS(unittest.TestCase):
    def setUp(self) -> None:
        self.NS = NS()
        
    def test_init(self):
        self.assertEqual(self.NS._SOI, {})
        
if __name__ == '__main__':
    unittest.main()