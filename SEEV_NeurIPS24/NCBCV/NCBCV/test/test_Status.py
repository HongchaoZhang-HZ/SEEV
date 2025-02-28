import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus

class TestNeuronStatus(unittest.TestCase):
    def test_init(self):
        neuron_status = NeuronStatus(1, 2, 0)
        self.assertEqual(neuron_status.layer, 1)
        self.assertEqual(neuron_status.neuron, 2)
        self.assertEqual(neuron_status.status, 0)

    def test_get_id(self):
        neuron_status = NeuronStatus(1, 2, 0)
        self.assertEqual(neuron_status.get_id(), [1, 2])

    def test_set_status(self):
        neuron_status = NeuronStatus(1, 2, 0)
        neuron_status.set_status(1)
        self.assertEqual(neuron_status.status, 1)

class TestNetworkStatus(unittest.TestCase):
    def setUp(self) -> None:
        self.network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
        self.NStatus = NetworkStatus(self.network)
    
    def test_init(self):
        self.assertEqual(self.NStatus.neuron_inputs, {})
        self.assertEqual(self.NStatus.network_status, {})
        self.assertEqual(self.NStatus.network_status_values, {})
        
    def test_set_layer_status(self):
        # Set the layer idx and status
        layer_idx = 1
        layer_status = [0, 1]  # Example layer status
        self.NStatus.set_layer_status(layer_idx, layer_status)
        layer_status_value = [neuron.status for neuron in self.NStatus.network_status]
        # Verify the layer status is set correctly
        self.assertEqual(layer_status_value, layer_status)
        
    def test_get_netstatus_from_input(self):
        input_size = self.network.layers[0].in_features
        input = torch.rand(input_size)
        self.NStatus.get_netstatus_from_input(input)

        # Check network status
        net_stat_values = self.NStatus.network_status_values
        net_stat = self.NStatus.network_status
        for layer_idx in range(len(net_stat)):
            layer_status = [item.status for item in net_stat[layer_idx]]
            self.assertEqual(layer_status, net_stat_values[layer_idx])

if __name__ == '__main__':
    unittest.main()
        
        