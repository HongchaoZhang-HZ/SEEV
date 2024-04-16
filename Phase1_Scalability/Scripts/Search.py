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
        self.verbose = False
        
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
        
        # prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
        # output constraints
        index_o = len(S.keys())-1
        prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), np.array(r_o[index_o]), x)
        
        lb, ub = self.BoundingBox_approxi(S)
        neighbour_neurons = {}
        for i in range(len(W_B)):
            neuron_list_layer = []
            for j in range(len(W_B[i])):
                eq_cons = prog.AddLinearEqualityConstraint(np.array(W_B[i][j]), np.array(r_B[i][j]), x)
                bounding_box = prog.AddBoundingBoxConstraint(lb, ub, x)
                result = Solve(prog)
                
                if result.is_success():
                    if self.verbose:
                        print(f"Neuron {i,j} is unstable at the boundary")
                    neuron_list_layer.append(j)
                prog.RemoveConstraint(eq_cons)
            neighbour_neurons[i] = neuron_list_layer
        if self.verbose:
            print(f"Number of neurons that are unstable at the boundary: {sum(len(item) for item in neighbour_neurons.values())}")
        return neighbour_neurons
    
    def Possible_S(self, ref_S, neighbour_neurons):
        '''
        Find the possible S that contains the unstable neurons
        Given ref_S as the activiation pattern we have for the network, 
        given the neurons that are unstable neighbour_neurons at the boundary of ref_S, 
        The function is to find the possible S that contains the unstable neurons.
        The possible S has its neuron contained indexed in the neighbour_neurons set to the opposite of the neurons of ref_S.
        Iterate over the items in the neighbour_neurons and generate the possible S that contains the unstable neurons as a list.
        output is a list of possible S that contains the unstable neurons.
        '''
        Possible_S = []
        for layer, neurons in neighbour_neurons.items():
            for neuron in neurons:
                neighbour_S = ref_S
                neighbour_S[layer][neuron] = -ref_S[layer][neuron]
                Possible_S.append(neighbour_S)
        return Possible_S
    
    def BFS(self, S):
        '''
        Breadth First Search
        Conduct breadth first search on the network to find the unstable neurons
        Initially, we start with the given set S, and then we find the neurons that are unstable at the boundary of S.
        To find the boundary of S, we conduct Filter_S_neighbour(S) to find the neurons that are unstable at the boundary of S.
        Then we add the neurons that are unstable at the boundary of S to the queue.
        We then iterate through the queue to find the neurons that are unstable at the boundary of the neurons that are unstable at the boundary of S.
        We continue this process until we find all the unstable neurons.
        '''
        queue = S
        unstable_neurons = set()
        boundary_set = set()
        while queue:
            current_set = queue.pop(0)
            res = self.identification_lp(current_set) 
            print(res.is_success())
            if res.is_success():
                boundary_set.add(current_set)
            unstable_neighbours = self.Filter_S_neighbour(current_set)
            
            for layer, neurons in unstable_neighbours.items():
                for neuron in neurons:
                    unstable_neurons.add((layer, neuron))
                    queue.append({**current_set, layer: [neuron]})
                    
        
        return boundary_set
    
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
    # print(Search.S_init)

    Search.Filter_S_neighbour(Search.S_init[0])
    Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Possible_S)
    unstable_neurons_set = Search.BFS(Possible_S)
    print(unstable_neurons_set)