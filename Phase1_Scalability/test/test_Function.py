import re
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

import unittest
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS

from Foundation.Function import RoA, LinearExp, solver_lp
from pydrake.solvers import MathematicalProgram, Solve

class TestFunctions(unittest.TestCase):
    def setUp(self):
        # Define a simple model and S for testing
        architecture = [('relu', 2), ('relu', 2), ('linear', 1)]
        self.model = NNet(architecture)
        self.model.layers[0].weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.model.layers[0].bias.data = torch.tensor([0.5, 0.5])
        self.model.layers[2].weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        self.model.layers[2].bias.data = torch.tensor([0.5, 0.5])
        self.model.layers[4].weight.data = torch.tensor([[1.0, 2.0]])
        self.model.layers[4].bias.data = torch.tensor([0.5])
        self.NStatus = NetworkStatus(self.model)

        # Generate random input using torch.rand for the model
        input_size = self.model.layers[0].in_features
        x = torch.zeros(input_size)
        self.NStatus.get_netstatus_from_input(x)
        self.S = self.NStatus.network_status_values

    def test_linearExp(self):
        # Call the function with the test inputs
        W_B, r_B, W_o, r_o = LinearExp(self.model, self.S)

        # Assert that the solution is correct
        # This will depend on the expected output of your function
        # Here is a placeholder for the assertion
        self.assertTrue(np.array_equal(W_B[0], np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(np.array_equal(r_B[0], np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(W_o[0], np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(np.array_equal(r_o[0], np.array([0.5, 0.5])))
        self.assertTrue(np.array_equal(W_B[1], 
                                       np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(np.array_equal(r_B[1], 
                                       np.array([[1.0, 2.0], [3.0, 4.0]]) @ np.array([0.5, 0.5]) + np.array([0.5, 0.5])))
        
    def test_RoA(self):
        # Call the function with the test inputs
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(2, "x")
        prog1 = RoA(prog, x, self.model, self.S)
        W_B, r_B, W_o, r_o = LinearExp(self.model, self.S)
        prog2 = RoA(prog, x, self.model, W_B=W_B, r_B=r_B)
        
        res1 = Solve(prog1)
        res2 = Solve(prog2)
        self.assertTrue(res1.is_success())
        self.assertTrue(res2.is_success())
        self.assertTrue(np.array_equal(res1.GetSolution(x), res2.GetSolution(x)))
        self.assertEqual(res1.get_optimal_cost(), res2.get_optimal_cost())
        
if __name__ == '__main__':
    unittest.main()
