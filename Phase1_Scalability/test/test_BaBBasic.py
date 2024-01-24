import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Modules.utils import *
from Scripts.NSBasic import NSBasic, NS
from Modules.NNet import NeuralNetwork as NNet
from Scripts.BaBBasic import QBasic, BaBBasic

class TestQBasic(unittest.TestCase):
    def setUp(self) -> None:
        network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.QBasic = QBasic(network)
    
    def test_init(self):
        self.assertEqual(self.QBasic.set_P, {})
        self.assertEqual(self.QBasic.set_N, {})
        self.assertEqual(self.QBasic.set_Z, {})
        for layer_idx in self.QBasic.set_U.keys():
            for neuron_idx in self.QBasic.set_U[layer_idx]:
                self.assertEqual(self.QBasic.set_U[layer_idx][neuron_idx].status, -2)


if __name__ == '__main__':
    unittest.main()
    