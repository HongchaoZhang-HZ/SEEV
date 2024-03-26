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
def LinearExp(S:dict) -> (np.array, np.array):
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
    # TODO: Implement the function to get the linear expression of the output of the ReLU NN
    
    # Get the number of layers in the NN
    num_layers = len(S)
    # TODO: get activation fucntion
    
    for keys, values in S.items():
        # Get the current layer
        layer = S[keys]
        
        # Get the activation function and the number of neurons in the layer
        activation, num_neurons = layer
        
        W = np.append(W, np.array([np.zeros(num_neurons)]))
        r = np.append(r, np.array([np.zeros(num_neurons)]))
        
    return (W, r)

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
    