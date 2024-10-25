import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Scripts.NSBasic import *
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
        
    def test_neuron_layer_is_in_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.set_U = {0: {0: neuron_status}}
        self.assertEqual(self.NSB.neuron_layer_is_in_set(self.NSB.set_U, neuron_status), True)
    
    def test_neuron_idx_is_in_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.set_U = {0: {0: neuron_status}}
        self.assertEqual(self.NSB.neuron_idx_is_in_layer(self.NSB.set_U[0], neuron_status), True)
        
    def test_is_in_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.set_U = {0: {0: neuron_status}}
        self.assertEqual(self.NSB.is_in_set('U', neuron_status), True)
    
    def test_add_to_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.add_to_set('U', neuron_status)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, -2).display())
        self.NSB.add_to_set('P', neuron_status)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, 1).display())
    
    def test_remove_from_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.set_U = {0: {0: neuron_status}}
        self.NSB.remove_from_set('U', neuron_status)
        self.assertEqual(self.NSB.set_U, {0: {}})
        
    def test_update_set(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.add_to_set('U', neuron_status)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, -2).display())
        neuron_status = NeuronStatus(0, 0, -1)
        self.NSB.update_set('U', neuron_status)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, -1).display())
    
    def test_move(self):
        neuron_status = NeuronStatus(0, 0, -2)
        self.NSB.add_to_set('U', neuron_status)
        self.assertEqual(self.NSB.set_U[0][0].display(), NeuronStatus(0, 0, -2).display())
        self.NSB.move('U', 'P', neuron_status)
        self.assertEqual(self.NSB.set_U, {0: {}})
        self.assertEqual(self.NSB.set_P[0][0].display(), NeuronStatus(0, 0, 1).display())
         
         
class TestNS(unittest.TestCase):
    def setUp(self) -> None:
        self.NS = NS()
        
    def test_init(self):
        self.assertEqual(self.NS.set_P, {})
        self.assertEqual(self.NS.set_N, {})
        self.assertEqual(self.NS.set_Z, {})
        self.assertEqual(self.NS.set_U, {})
        
    def test_network(self):
        network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.NS.get_network(network)
        self.assertEqual(self.NS.network, network)
        self.assertEqual(self.NS.set_U[0][0].display(), NeuronStatus(0, 0, -2).display())
        self.assertEqual(self.NS.set_U[1][31].display(), NeuronStatus(1, 31, -2).display())
        self.assertEqual(self.NS.set_U[2][0].display(), NeuronStatus(2, 0, -2).display())
  
    def test_get_NS_input(self):
        network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.NS.init_NS(network)
        input_size = self.NS.network.layers[0].in_features
        input = torch.rand(input_size)
        dict_NSV = self.NS.get_NS_input(input, 'NSV')
        dict_NS = self.NS.get_NS_input(input, 'NS')
        
        # Check network status
        net_stat_values = dict_NSV
        net_stat = dict_NS
        for layer_idx in range(len(net_stat)):
            layer_status = [item.status for item in net_stat[layer_idx]]
            self.assertEqual(layer_status, net_stat_values[layer_idx])

    
if __name__ == '__main__':
    unittest.main()