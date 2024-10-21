import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.NNet import *

class TestNeuralNetwork(unittest.TestCase):
    def test_init(self):
        # Create an instance of the NeuralNetwork class with a given architecture
        # Assuming the network takes input of 64 and outputs size 1
        architecture = [('relu', 64), ('relu', 32), ('linear', 1)]
        # validate the architecture
        model = NeuralNetwork(architecture)
        # validate the model
        self.assertEqual(model.layers[0].in_features, 64)
        self.assertEqual(model.layers[2].out_features, 32)
        self.assertEqual(model.layers[4].out_features, 1)
        
        # validate activation functions
        self.assertIsInstance(model.layers[1], nn.ReLU)
        self.assertIsInstance(model.layers[3], nn.ReLU)
        self.assertIsInstance(model.layers[5], nn.Identity)
        
        # validate the device
        self.assertEqual(model.device, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
    def test_forward(self):
        # Create an instance of the NeuralNetwork class
        architecture = [('relu', 64), ('relu', 32), ('linear', 1)]
        model = NeuralNetwork(architecture)
        # validate the forward method
        x = torch.randn(64)
        # validate the output
        self.assertEqual(model.forward(x).shape, torch.Size([1]))
        # validate the device
        self.assertEqual(model.forward(x).device, model.device)

if __name__ == '__main__':
    unittest.main()