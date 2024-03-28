from pydrake.solvers import MathematicalProgram, Solve
import numpy as np
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus

# Given a linear expression of a ReLU NN (Activation set $S$), 
# return a set of linear constraints to formulate $\mathcal{X}(S)$

# Region of Activation (RoA) is the set of points that are activated by a ReLU NN
def RoA(prog:MathematicalProgram, x, model,
        S:dict=None, W_B:dict=None, r_B:dict=None) -> MathematicalProgram:
    # check if S is provided
    if S is None:
        # check if W_B, r_B are provided
        if W_B is None or r_B is None:
            raise ValueError("Activation set S or (W_B, r_B) are not provided")
    else:
        # if not, compute the linear expression of the output of the ReLU NN
        W_B, r_B, _, _ = LinearExp(model, S)
        
    assert len(W_B) == len(r_B), "W_B and r_B must have the same amount of layers"
    # stack W_B and r_B constraints
    for keys, layer_info in W_B.items():
        if keys == 0:
            cons_W = layer_info
            cons_r = r_B[keys]
        else:
            cons_W = np.vstack([cons_W, layer_info])
            cons_r = np.hstack([cons_r, r_B[keys]])
    
    # Add linear constraints to the MathematicalProgram
    linear_constraint = prog.AddLinearConstraint(A= cons_W, lb=-cons_r, 
                                                 ub=np.inf*np.ones(len(cons_r)), vars=x)
    print(linear_constraint)
    return prog

# Given a activation set $S$, return the linear expression of the output of the ReLU NN
def LinearExp(model, S:dict) -> (dict, dict, dict, dict):
    # Input: S: Activation set of a ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    W_list = []
    r_list = []
    para_list = list(model.state_dict())
    i = 0
    while i < (len(para_list)):
        weight = model.state_dict()[para_list[i]]
        i += 1
        bias = model.state_dict()[para_list[i]]
        i += 1
        W_list.append(weight)
        r_list.append(bias)
        
    W_B = dict()
    r_B = dict()
    W_o = dict()
    r_o = dict()
    for keys, layer_info in S.items():
        # Get the current activation layer
        layer_act_list = torch.relu(torch.tensor(layer_info))
        layer_act_output_array = np.array(layer_act_list)
        layer_act_bound_array = np.array(layer_info)
        
        # compute output equivalent weight and bias for each layer
        W_o_layer = np.multiply(np.expand_dims(layer_act_output_array,-1), W_list[keys])
        r_o_layer = np.multiply(layer_act_output_array, r_list[keys])
        
        # compute boundary weight and bias for each layer
        W_B_layer = np.multiply(np.expand_dims(layer_act_bound_array,-1), W_list[keys])
        r_B_layer = np.multiply(layer_act_output_array, r_list[keys])
        # add boundary condition to W_B and r_B
        if keys == 0:
            W_B[keys] = np.array(W_B_layer)
            r_B[keys] = np.array(r_B_layer)
            W_o[keys] = np.array(W_o_layer)
            r_o[keys] = np.array(r_o_layer)
        else:
            W_o[keys] = np.matmul(W_o_layer, W_o[keys-1])
            r_o[keys] = np.matmul(W_o_layer, r_o[keys-1]) + np.array(r_o_layer)
            
            W_B[keys] = np.matmul(W_B_layer, W_o[keys-1])
            r_B[keys] = np.matmul(W_B_layer, r_o[keys-1]) + np.array(r_B_layer)
        
    return W_B, r_B, W_o, r_o

def solver_lp(model, S):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    x = prog.NewContinuousVariables(2, "x")
    prog = RoA(prog, x, model, S)
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(S)
    print(W_B, r_B, W_o, r_o)
    
    # Output layer index
    index_o = len(S.keys())-1
    # Add linear constraints
    prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
    # Add cost function
    # QC = prog.AddQuadraticCost(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]))
    
    # Now solve the program.
    result = Solve(prog)
    print(f"Is solved successfully: {result.is_success()}")
    print(f"x optimal value: {result.GetSolution(x)}")
    print(f"optimal cost: {result.get_optimal_cost()}") 
    
if __name__ == "__main__":

    architecture = [('relu', 2), ('relu', 32), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    random_input = torch.rand(input_size)
    x = random_input
    NStatus.get_netstatus_from_input(x)
    S = NStatus.network_status_values
    solver_lp(model, S)