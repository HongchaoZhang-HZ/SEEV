import re
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
        S:dict=None, W_B:dict=None, r_B:dict=None, SSpace=[-2,2]) -> MathematicalProgram:
    # check if S is provided
    if S is None:
        # check if W_B, r_B are provided
        if W_B is None or r_B is None:
            raise ValueError("Activation set S or (W_B, r_B) are not provided")
    else:
        # if not, compute the linear expression of the output of the ReLU NN
        W_B, r_B, _, _ = LinearExp(model, S)
    final_layer_idx = len(W_B.keys())
    assert len(W_B) == len(r_B), "W_B and r_B must have the same amount of layers"
    # stack W_B and r_B constraints
    for keys, layer_info in W_B.items():
        if keys == 0:
            cons_W = layer_info
            cons_r = r_B[keys]
        elif keys == final_layer_idx:
            break
        else:
            cons_W = np.vstack([cons_W, layer_info])
            cons_r = np.hstack([cons_r, r_B[keys]])
    
    # Add linear constraints to the MathematicalProgram
    linear_constraint = prog.AddLinearConstraint(A= cons_W, lb=-cons_r, 
                                                 ub=np.inf*np.ones(len(cons_r)), vars=x)
    boundingbox = prog.AddBoundingBoxConstraint(SSpace[0], SSpace[1], x)
    # print(linear_constraint)
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
        r_B_layer = np.multiply(layer_act_bound_array, r_list[keys])
        # add boundary condition to W_B and r_B
        if keys == 0:
            W_B[keys] = np.array(W_B_layer)
            r_B[keys] = np.array(r_B_layer)
            W_o[keys] = np.array(W_o_layer)
            r_o[keys] = np.array(r_o_layer)
        elif keys == len(S.keys())-1:
            W_o[keys] = np.array(np.matmul(W_list[keys], W_o[keys-1]))
            r_o[keys] = np.array(np.matmul(W_list[keys], r_o[keys-1]) + np.array(r_list[keys]))
        else:
            W_o[keys] = np.array(np.matmul(W_o_layer, W_o[keys-1]))
            r_o[keys] = np.array(np.matmul(W_o_layer, r_o[keys-1])) + np.array(r_o_layer)
            
            W_B[keys] = np.array(np.matmul(W_B_layer, W_o[keys-1]))
            r_B[keys] = np.array(np.matmul(W_B_layer, r_o[keys-1])) + np.array(r_B_layer)
        
    return W_B, r_B, W_o, r_o

def HyperCube_Approximation(model, S):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = next(model.children())[0].in_features
    x = prog.NewContinuousVariables(dim, "x")
    
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
    
    # print(W_B, r_B, W_o, r_o)
    
    # Output layer index
    index_o = len(S.keys())-1
    # Add linear constraints
    prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)
    
    HyperCube = {}
    for i in range(dim):
        lb = prog.AddCost(x[i])
        result = Solve(prog)
        lb_solution = result.GetSolution(x[i])
        prog.RemoveCost(lb)
        ub = prog.AddCost(-x[i])
        result = Solve(prog)
        ub_solution = result.GetSolution(x[i])
        prog.RemoveCost(ub)
        # print('Lower Bound:', lb_solution)
        # print('Upper Bound:', ub_solution)
        # HyperCube[i] = [round(lb_solution, round_tol), 
        #                 round(ub_solution, round_tol)]
        HyperCube[i] = [lb_solution, ub_solution]
        
    # print('HyperCube:', HyperCube)

    return HyperCube

def solver_lp(model, S):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = next(model.children())[0].in_features
    x = prog.NewContinuousVariables(dim, "x")
    
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
    
    # Output layer index
    index_o = len(S.keys())-1
    # Add linear constraints
    prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)
    
    # Now solve the program.
    result = Solve(prog)

    print('check result:', np.matmul(W_o[index_o], result.GetSolution(x)) + r_o[index_o], W_o[index_o], r_o[index_o])
    print('ref_result:', model.forward(torch.tensor(result.GetSolution(x)).float()))
    return result
    
if __name__ == "__main__":

    # architecture = [('relu', 2), ('relu', 32), ('relu', 32), ('linear', 1)]
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # model = NNet(architecture)
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    random_input = torch.rand(input_size)
    NStatus.get_netstatus_from_input(random_input)
    S = NStatus.network_status_values
    solver_lp(model, S)
    HyperCube_Approximation(model, S)