import sys, os

from numpy import ubyte
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from Scripts.SearchInit import *
from Scripts.SearchInit import SearchInit
from Modules.Function import HyperCube_Approximation

class Search(SearchInit):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        self.model = model
        self.case = case
        self.NStatus = NetworkStatus(model)
        
    def Specify_point(self, safe_point:torch.tensor, unsafe_point:torch.tensor):
        S_init = Search.initialization(safe_point, unsafe_point)
        self.S_init = S_init
        
    def Hypercube_approxi(self, S) -> dict:
        HyperCube =  HyperCube_Approximation(self.model, S)
        return HyperCube
    
    def BoundingBox_approxi(self, S) -> (list, list):
        cube = self.Hypercube_approxi(S)
        lb_list = []
        ub_list = []
        for dim in range(len(cube)):
            lb = cube[dim][0]
            lb_list.append(lb)
            ub = cube[dim][1]
            ub_list.append(ub)
        return lb_list, ub_list
    
    def Filter_S_neighbour(self, S):
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(self.dim, "x")
        # Add linear constraints
        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        
        prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
        # output constraints
        index_o = len(S.keys())-1
        prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), np.array(r_o[index_o]), x)
        
        # lb, ub = self.BoundingBox_approxi(S)
        neibour_neurons = {}
        for i in range(len(W_B)):
            neuron_list_layer = []
            for j in range(len(W_B[i])):
                eq_cons = prog.AddLinearEqualityConstraint(np.array(W_B[i][j]), np.array(r_B[i][j]), x)
                # bounding_box = prog.AddBoundingBoxConstraint(lb, ub, x)
                result = Solve(prog)
                
                if result.is_success():
                    print(f"Neuron {i,j} is unstable at the boundary")
                    neuron_list_layer.append(j)
                prog.RemoveConstraint(eq_cons)
            neibour_neurons[i] = neuron_list_layer
        print(f"Number of neurons that are unstable at the boundary: {sum(len(item) for item in neibour_neurons.values())}")
        return neibour_neurons
    
if __name__ == "__main__":
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search = Search(model)
    # (0.5, 1.5), (0, -1)
    Search.Specify_point(torch.tensor([[[0.5, 1.8]]]), torch.tensor([[[0, -1]]]))
    print(Search.S_init)

    Search.Filter_S_neighbour(Search.S_init[0])
    