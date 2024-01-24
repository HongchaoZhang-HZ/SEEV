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
    
    def test_init_mask(self):
        mask = self.QBasic.init_mask()
        for layer_idx in range(len(self.QBasic._net_size)):
            self.assertEqual(mask[layer_idx].shape[0], self.QBasic._net_size[layer_idx])
            self.assertEqual(mask[layer_idx].dtype, torch.float32)

    def test_update_Q_frrom_NS(self):
        input_size = self.QBasic._NStatus.network.layers[0].in_features
        input = torch.rand(input_size)
        self.QBasic._NStatus.get_netstatus_from_input(input)
        self.QBasic.update_Q_from_NS()
        # N[0][0]
        N_Status = self.QBasic._NStatus.network_status[0][0]
        if N_Status.status == 1:
            self.assertEqual(self.QBasic.set_P[0][0], N_Status)
        elif N_Status.status == -1:
            self.assertEqual(self.QBasic.set_N[0][0], N_Status)
        elif N_Status.status == 0:
            self.assertEqual(self.QBasic.set_Z[0][0], N_Status)
        else:
            self.assertEqual(self.QBasic.set_U[0][0], N_Status)
    
if __name__ == '__main__':
    unittest.main()
    