import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Modules.NCBF import NCBF
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
from Scripts.BaBBasic import QBasic, BaBBasic
from BaB.BaB_ReLU import ReLU_Q

# class TestReLU_Q(unittest.TestCase):
#     def setUp(self) -> None:
#         network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
#         self.ReLU_Q = ReLU_Q(network)
        
    # def test_init(self):
    #     self.assertEqual(self.ReLU_Q.set_P, {})
    #     self.assertEqual(self.ReLU_Q.set_N, {})
    #     self.assertEqual(self.ReLU_Q.set_Z, {})
    #     for layer_idx in self.ReLU_Q.set_U.keys():
    #         for neuron_idx in self.ReLU_Q.set_U[layer_idx]:
    #             self.assertEqual(self.ReLU_Q.set_U[layer_idx][neuron_idx].status, -2)
    
    # def test_init_mask(self):
    #     mask = self.ReLU_Q.init_mask()
    #     for layer_idx in range(len(self.ReLU_Q._net_size)):
    #         self.assertEqual(mask[layer_idx].shape[0], self.ReLU_Q._net_size[layer_idx])
    #         self.assertEqual(mask[layer_idx].dtype, torch.int8)

    # def test_update_Q_frrom_NS(self):
    #     input_size = self.ReLU_Q._NStatus.network.layers[0].in_features
    #     input = torch.rand(input_size)
    #     self.ReLU_Q._NStatus.get_netstatus_from_input(input)
    #     self.ReLU_Q.update_Q_from_NS()
    #     # N[0][0]
    #     N_Status = self.ReLU_Q._NStatus.network_status[0][0]
    #     if N_Status.status == 1:
    #         self.assertEqual(self.ReLU_Q.set_P[0][0], N_Status)
    #     elif N_Status.status == -1:
    #         self.assertEqual(self.ReLU_Q.set_N[0][0], N_Status)
    #     elif N_Status.status == 0:
    #         self.assertEqual(self.ReLU_Q.set_Z[0][0], N_Status)
    #     else:
    #         self.assertEqual(self.ReLU_Q.set_U[0][0], N_Status)
    
    # def test_update_mask(self):
    #     self.test_update_Q_frrom_NS()
    #     self.ReLU_Q.update_mask()
    #     for layer_idx in range(len(self.ReLU_Q._net_size)):
    #         self.assertEqual(self.ReLU_Q.mask[layer_idx].dtype, torch.int8)
    #         self.assertEqual(self.ReLU_Q.mask[layer_idx].tolist(), 
    #                          self.ReLU_Q._NStatus.network_status_values[layer_idx])