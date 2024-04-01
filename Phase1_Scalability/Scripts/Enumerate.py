import sys, os

from torch import R
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
import PARA as p
from Modules.Function import *

# class Enumerate:
#     def __init__(self) -> None:
#         self.zero_tol = p.zero_tol
#         pass
    
def test():
    # Define a simple model and S for testing
    architecture = [('relu', 2), ('relu', 2), ('linear', 1)]
    model = NNet(architecture)
    model.layers[0].weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model.layers[0].bias.data = torch.tensor([0.5, 0.5])
    model.layers[2].weight.data = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    model.layers[2].bias.data = torch.tensor([0.5, 0.5])
    model.layers[4].weight.data = torch.tensor([[1.0, 2.0]])
    model.layers[4].bias.data = torch.tensor([0.5])
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    x = torch.zeros(input_size)
    NStatus.get_netstatus_from_input(x)
    S = NStatus.network_status_values
    
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
    output_cons = prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
    
    for i in range(len(W_B)):
        for j in range(len(W_B[i])):
            eq_cons = prog.AddLinearEqualityConstraint(np.array(W_B[i][j]), np.array(r_B[i][j]), x)
            result = Solve(prog)
            print(f"Is solved successfully: {result.is_success()}")
            if result.is_success():
                print(f"Neuron {i} is unstable at the boundary")
            prog.RemoveConstraint(eq_cons)

def gridify(state_space, shape, cell_length):
    for i in range(len(shape)):
        dn[i] = torch.linspace(state_space[0][0] + cell_length / 2, state_space[0][1] - cell_length / 2, shape[0])
    data = torch.stack(torch.meshgrid(*dn),dim=-1).reshape(*shape, -1)
    return data

def grid_intialization(input_size, model, NStatus, S_init_Set, m = 10):
    input_set = gridify(state_space=[[-2,2],[-2,2]], shape=[10, 10], cell_length=0.1)
    for i in range(m):
        NStatus.get_netstatus_from_input(input_set[i])
        S = NStatus.network_status_values
        res = solver_lp(model, S)
        if res.is_success():
            print(f"Random initialization {i} is successful")
            S_init_Set.add(tuple(tuple(items) for keys, items in S.items()))
    return S_init_Set

# TODO: FIX THIS this function does not provide a good initialization
def rdm_intialization(input_size, model, NStatus, S_init_Set, m = 10):
    input_set = [torch.rand(input_size) * 4 - 2 for _ in range(m)]
    for i in range(m):
        NStatus.get_netstatus_from_input(input_set[i])
        S = NStatus.network_status_values
        res = solver_lp(model, S)
        if res.is_success():
            print(f"Random initialization {i} is successful")
            S_init_Set.add(tuple(tuple(items) for keys, items in S.items()))
    return S_init_Set

def test_with_model():
    # Define a simple model and S for testing
    architecture = [('relu', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    model.state_dict(torch.load("Phase1_Scalability/darboux_1_32.pt"))
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    S_init_Set = set()
    while len(S_init_Set)==0:
        S_init_Set = rdm_intialization(input_size, model, NStatus, S_init_Set, m = 100)
    S = S_init_Set.pop()
    
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
    output_cons = prog.AddLinearEqualityConstraint(np.array(W_o[index_o]).flatten(), np.array(r_o[index_o]), x)
    
    for i in range(len(W_B)):
        for j in range(len(W_B[i])):
            eq_cons = prog.AddLinearEqualityConstraint(np.array(W_B[i][j]), np.array(r_B[i][j]), x)
            result = Solve(prog)
            print(f"Is solved successfully: {result.is_success()}")
            if result.is_success():
                print(f"Neuron {i} is unstable at the boundary")
            prog.RemoveConstraint(eq_cons)

    
if __name__ == "__main__":
    test_with_model()