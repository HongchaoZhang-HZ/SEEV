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
def RoA(S:dict, prog:MathematicalProgram, x):
    # Initialize the set of constraints
    X = []
    
    W, r = LinearExp(S)
    
    # Return the set of constraints
    return prog

def activated_weight_bias_ml(model,activated_set,num_neuron):
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
    # compute the activated weight of the layer
    for l in range(2):
        # compute region/boundary weight


        if l == 0:
            W_l = torch.mul(activated_set[num_neuron*l:num_neuron*(l+1)], W_list[l])
            r_l = torch.mul(activated_set[num_neuron*l:num_neuron*(l+1)], torch.reshape(r_list[l], [len(r_list[l]), 1]))
            W_a = W_l
            r_a = r_l
            W_i = W_list[l] - W_l
            r_i = -torch.reshape(r_list[l], [len(r_list[l]), 1]) + r_l
        else:
            W_pre = W_list[l] @ W_l
            r_pre = W_list[l] @ r_l + r_list[l].reshape([len(r_list[l]), 1])
            W_l = activated_set[num_neuron*l:num_neuron*(l+1)]*W_pre
            r_l = activated_set[num_neuron*l:num_neuron*(l+1)]*r_pre
            W_a = torch.vstack([W_a, W_l])
            r_a = torch.vstack([r_a, r_l])
            W_i = torch.vstack([W_i, W_pre - W_l])
            r_i = torch.vstack([r_i, -torch.reshape(r_pre, [len(r_pre), 1]) + r_l])
        B_act = [W_a, r_a]  # W_a x <= r_a
        B_inact = [W_i, r_i]  # W_a x <= r_a
    # W_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], W_l))  # compute \overline{W}(S)
    # r_overl = torch.matmul(W_list[-1], torch.matmul(W_list[-2], r_l) + r_list[-2].reshape([num_neuron,1])) + r_list[-1]  # compute \overline{r}(S)
    W_overl = torch.matmul(W_list[-1], W_l)  # compute \overline{W}(S)
    r_overl = torch.matmul(W_list[-1], r_l) + r_list[-1]  # compute \overline{r}(S)
    return W_overl, r_overl, B_act, B_inact

# Given a activation set $S$, return the linear expression of the output of the ReLU NN
def LinearExp(S:dict) -> (dict, dict, dict, dict):
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
            W_B[keys] = W_B_layer
            r_B[keys] = r_B_layer
            W_o[keys] = W_o_layer
            r_o[keys] = r_o_layer
        else:
            W_o[keys] = np.matmul(W_o_layer, W_o[keys-1])
            r_o[keys] = np.matmul(W_o_layer, r_o[keys-1]) + r_o_layer
            
            W_B[keys] = np.matmul(W_B_layer, W_o[keys-1])
            r_B[keys] = np.matmul(W_B_layer, r_o[keys-1]) + r_B_layer
        
    return W_B, r_B, W_o, r_o

def solver_lp():
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    x = prog.NewContinuousVariables(2, "x")

    # Add a symbolic linear expression as the cost.
    cost1 = prog.AddLinearCost(x[0] + 3 * x[1] + 2)
    # Print the newly added cost
    print(cost1)
    # The newly added cost is stored in prog.linear_costs().
    print(prog.linear_costs()[0])
    
    
if __name__ == "__main__":
    solver_lp()

    architecture = [('relu', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    random_input = torch.rand(input_size)
    x = random_input
    NStatus.get_netstatus_from_input(x)
    S = NStatus.network_status_values
    W_B, r_B, W_o, r_o = LinearExp(S)
    print(W_B, r_B, W_o, r_o)