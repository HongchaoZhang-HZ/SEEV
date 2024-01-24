import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Scripts.NSBasic import *
from Modules.utils import *
from Scripts.NSBasic import NSBasic, NS
from Modules.NNet import NeuralNetwork as NNet
from Scripts.BaBBasic import QBasic, BaBBasic

class TestQBasic(unittest.TestCase):
    def setUp(self) -> None:
        network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.QBasic = BaBBasic(network)
    
    def test_init(self):
        self.assertEqual(self.QBasic.NStatus.neuron_inputs, {})
        self.assertEqual(self.QBasic.NStatus.network_status, {})
        self.assertEqual(self.QBasic.NStatus.network_status_values, {})
        self.assertEqual(self.QBasic.NS.set_P, {})
        self.assertEqual(self.QBasic.NS.set_N, {})
        self.assertEqual(self.QBasic.NS.set_Z, {})
        self.assertEqual(self.QBasic.NS.set_U, {})
