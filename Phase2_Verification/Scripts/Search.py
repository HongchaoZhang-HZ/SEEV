import sys, os
import copy
from numpy import ubyte
from tomlkit import item
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from itertools import combinations
from collections import deque
from Scripts.SearchInit import *
from Scripts.SearchInit import SearchInit
from Modules.Function import HyperCube_Approximation
from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm
class Search(SearchInit):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        self.model = model
        self.case = case
        self.NStatus = NetworkStatus(model)
        self.verbose = False
        
    def Specify_point(self, safe_point:torch.tensor, unsafe_point:torch.tensor):
        S_init = self.initialization(input_safe=safe_point, input_unsafe=unsafe_point)
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
    
    def IBP_S_neighbour(self, S):
        # prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        # x = prog.NewContinuousVariables(self.dim, "x")
        # Add linear constraints
        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        
        # prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
        # output constraints
        index_o = len(S.keys())-1
        # prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), np.array(r_o[index_o]), x)
        
        lb, ub = self.BoundingBox_approxi(S)
        neighbour_neurons = {}
        for layer_idx, layer in enumerate(self.model.layers):
            if layer_idx % 2 != 0 or layer_idx == 0:
                continue
            # conduct crown-IBP in auto-lirpa to enumerate all possible neurons that are unstable at the boundary
            target_model = torch.nn.Sequential(self.model.layers[0:layer_idx])
            
            # Define the forward function for the target_model
            def forward(self, x):
                for module in self:
                    x = module(x)
                return x
            
            # Set the forward function for the target_model
            target_model.forward = forward.__get__(target_model)
            # NotImplementedError: Module [ModuleList] is missing the required "forward" function
            
            # we iterate over all layers. In each layer, we compare the sign of the neurons outputs of lower and upper bounds.
            ibp_input = torch.tensor((np.array(lb)+np.array(ub))/2)
            # Wrap the model with auto_LiRPA.
            ibp_model = BoundedModule(target_model, ibp_input)
            # Define perturbation. Here we add Linf perturbation to input data.
            ptb = PerturbationLpNorm(norm=np.inf, eps=(ub-lb)/2)
            # Make the input a BoundedTensor with the pre-defined perturbation.
            ibp_input = BoundedTensor(ibp_input, ptb)
            # Regular forward propagation using BoundedTensor works as usual.
            prediction = ibp_model(ibp_input)
            # Compute LiRPA bounds using the backward mode bound propagation (CROWN).
            lb, ub = ibp_model.compute_bounds(x=(ibp_input,), method="backward")
        if self.verbose:
            print(f"Number of neurons that are unstable at the boundary: {sum(len(item) for item in neighbour_neurons.values())}")
        return neighbour_neurons
    
    def Filter_S_neighbour(self, S):
        # prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        # x = prog.NewContinuousVariables(self.dim, "x")
        # Add linear constraints
        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        
        # prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
        # output constraints
        index_o = len(S.keys())-1
        # prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), np.array(r_o[index_o]), x)
        
        lb, ub = self.BoundingBox_approxi(S)
        neighbour_neurons = {}
        for i in range(len(W_B)):
            neuron_list_layer = []
            for j in range(len(W_B[i])):
                # TODO: check if the neuron is unstable
                prog = MathematicalProgram()
                x = prog.NewContinuousVariables(self.dim, "x")
                zero_cons = prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)
                eq_cons = prog.AddLinearEqualityConstraint(np.array(W_B[i][j]), -np.array(r_B[i][j]), x)
                # prog = RoA(prog, x, model, S=None, W_B=W_B, r_B=r_B)
                bounding_box = prog.AddBoundingBoxConstraint(lb, ub, x)
                result = Solve(prog)
                # print(result.is_success())
                
                if result.is_success():
                    if self.verbose:
                        print(f"Neuron {i,j} is unstable at the boundary")
                    neuron_list_layer.append(j)
                # prog.RemoveConstraint(eq_cons)
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
                neighbour_S = copy.deepcopy(ref_S)
                neighbour_S[layer][neuron] = -ref_S[layer][neuron]
                Possible_S.append(copy.deepcopy(neighbour_S))
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
        queue = deque() 
        queue.append(S)
        # queue = S
        unstable_neurons = set()
        boundary_set = set()
        boundary_list = []
        previous_set = None
        pair_wise_hinge = []
        while queue:
            current_set = queue.popleft()
            # current_set = queue.pop(0)
            res = solver_lp(self.model, current_set) 
            # res = solver_lp(self.model, current_set)
            # print(res.is_success())
            if res.is_success():
                # add the current_set to the boundary_set (visited set)
                hashable_d = {k: tuple(v) for k, v in current_set.items()}
                tuple_representation = tuple(sorted(hashable_d.items()))
                if tuple_representation in boundary_set:
                    continue
                if previous_set is not None:
                    pair_wise_hinge.append([previous_set, current_set])
                boundary_set.add(tuple_representation)
                boundary_list.append(current_set)
                
                # finding neighbours
                unstable_neighbours = self.Filter_S_neighbour(current_set)
                # unstable_neighbours = self.IBP_S_neighbour(current_set)
                # unstable_neighbours = {0:[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], 1:[]}
                # unstable_neighbours = {0:[23, 30], 1:[]}
                unstable_neighbours_S = self.Possible_S(current_set, unstable_neighbours)
                
                # check repeated set
                # for idx in range(len(unstable_neighbours_S)):
                for item in unstable_neighbours_S:    
                    hashable_u = {k: tuple(v) for k, v in item.items()}
                    tuple_representation = tuple(sorted(hashable_u.items()))
                    if tuple_representation in boundary_set:
                        # remove idx from unstable_neighbours_S
                        unstable_neighbours_S.remove(item)
                
                # add the unstable neighbours to the queue
                queue.extend(unstable_neighbours_S)
            else:
                continue
            previous_set = current_set
        return boundary_list, pair_wise_hinge
    
    def hinge_lp(self, S, neighbor_seg_list):
        # For each item in the neighbor_seg_list, we need to find the hinge point.
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        # Add linear constraints
        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        prog = RoA(prog, x, self.model, S=None, W_B=W_B, r_B=r_B)
        
        # Output layer index
        index_o = len(S.keys())-1
        # Add linear constraints
        prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)

        # If the neighbor_seg_list is not empty, we iterate over the items in the neighbor_seg_list to find the hinge point.
        if len(neighbor_seg_list) != 0:
            for neighbor_seg in neighbor_seg_list:
                # TODO: This is a quick fix but we need to identify the bug and fix it.
                while isinstance(neighbor_seg, list):
                    neighbor_seg = neighbor_seg[0]
                # find the hinge point
                prog = RoA(prog, x, self.model, S=neighbor_seg)
        res = Solve(prog)
        return res.is_success()
    
    def hinge_post_identification(self, S, hinge_list, prior_seg_list, post_seg_list, hinge_prior_seg_list):
        hinge_post_seg_list = [] # this list stores the post_seg that are feasible
        for post_seg in post_seg_list:
            # find the hinge point
            tempt_prior_list = [prior_seg for prior_seg in prior_seg_list]
            tempt_prior_post_list = copy.deepcopy(tempt_prior_list)
            tempt_prior_post_list.append(post_seg)
            tempt_S_prior_post_list = [S]
            tempt_S_prior_post_list.extend(copy.deepcopy(tempt_prior_post_list))
            if tempt_S_prior_post_list in [hinge_list]:
                continue
            feasibility_flag = self.hinge_lp(S, tempt_prior_post_list)
            if feasibility_flag:
                hinge_list.extend(tempt_S_prior_post_list)
                hinge_post_seg_list.append(post_seg)
                hinge_prior_seg_list.extend(tempt_prior_list)
        
        
        if len(hinge_post_seg_list) > 1:
            space = np.min([len(prior_seg_list) + len(post_seg_list) + 1, self.dim-2])
            for r in range(1, np.min([len(hinge_post_seg_list) + 1, space])):
                if r+1+len(hinge_prior_seg_list) >= 6:
                    print("r is greater than 6")
                print("post r is ", r)
                for combo in combinations(hinge_post_seg_list, r):
                    tempt_list = [prior_seg for prior_seg in prior_seg_list]
                    tempt_list = tempt_list + list(combo)
                    # for item in list(combo):
                    #     tempt_list.append(item)
                    feasibility_flag = self.hinge_lp(S, tempt_list)
                    if feasibility_flag:
                        hinge_temp_list = copy.deepcopy(tempt_list)
                        hinge_temp_list.append(S)
                        hinge_list.extend(hinge_temp_list)
        return hinge_list, hinge_prior_seg_list, hinge_post_seg_list
    
    def hinge_identification(self, S, prior_seg_list, post_seg_list):
        # For each item in the prior_seg_list, and post_seg_list, we need to find the hinge point.
        # If the prior_seg_list is not empty, we iterate over the items in the prior_seg_list to find the hinge point.
        hinge_list = []
        if len(prior_seg_list) != 0 and len(post_seg_list) != 0:
            hinge_prior_seg_list = [] # this list stores the prior_seg that are feasible
            for prior_seg in prior_seg_list:
                # If the post_seg_list is not empty, we iterate over the items in the post_seg_list to find the hinge point.
                hinge_post_seg_list = [] # this list stores the post_seg that are feasible
                hinge_list, hinge_prior_seg_list, hinge_post_seg_list = self.hinge_post_identification(S, hinge_list, [prior_seg], post_seg_list, hinge_prior_seg_list)
            
            if len(hinge_prior_seg_list) != 0:
                space = np.min([len(prior_seg_list) + len(post_seg_list) + 1, self.dim-2])
                for r in range(1, np.min([len(hinge_prior_seg_list) + 1, space])):
                    if r >= self.dim-2:
                        break
                    for prior_seg in combinations(hinge_prior_seg_list, r):
                        hinge_list, hinge_prior_seg_list, hinge_post_seg_list = self.hinge_post_identification(S, hinge_list, list(prior_seg), post_seg_list, hinge_prior_seg_list)
        return hinge_list

    def hinge_search(self, boundary_list, pair_wise_hinge):
        # For low dim cases the pair_wise_hinge is small and maybe a loop. Therefore it is easy to enumarate nearby hinge hyperplane. 
        # For high dim cases, the pair_wise_hinge is large and maybe not be a loop. Therefore, a search is needed to find combinations. 
        # The overall design of the search is based on exhaustive search for completeness. 
        # The enumeration of neighboring hyperplanes is based on the pair_wise_hinge and simple search.
        hinge_list = []
        for mid_linear_segment in boundary_list:
            # for each linear segment, find the hinge hyperplane nearby
            prior_seg_list = []
            post_seg_list = []
            for pair in pair_wise_hinge:
                # find prior segment
                if pair[1] == mid_linear_segment:
                    prior_seg_list.append(pair[0])
                # find post segment
                if pair[0] == mid_linear_segment:
                    post_seg_list.append(pair[1])
            # check if prior and post segment sets are empty sets
            if len(prior_seg_list) == 0 and len(post_seg_list) == 0:
                continue
            # check if intersections happens
            ho_hinge_list = self.hinge_identification(mid_linear_segment, prior_seg_list, post_seg_list)
            if len(ho_hinge_list) != 0:
                hinge_list.append(ho_hinge_list)
        return hinge_list
            
        
        
if __name__ == "__main__":
    architecture = [('linear', 2), ('relu', 64), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/models/darboux_1_64.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search = Search(model)
    # (0.5, 1.5), (0, -1)
    Search.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0]]]))
    # print(Search.S_init)

    # Search.Filter_S_neighbour(Search.S_init[0])
    # Possible_S = Search.Possible_S(Search.S_init[0], Search.Filter_S_neighbour(Search.S_init[0]))
    # print(Search.Filter_S_neighbour(Search.S_init[0]))
    unstable_neurons_set, pair_wise_hinge = Search.BFS(Search.S_init[0])
    # unstable_neurons_set = Search.BFS(Possible_S)
    # print(unstable_neurons_set)
    print(len(unstable_neurons_set))
    print(len(pair_wise_hinge))
    
    ho_hinge = Search.hinge_search(unstable_neurons_set, pair_wise_hinge)
    print(len(ho_hinge))
    # test_S[0]=[-1, 1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1]
    # test_S[1] = [-1]
    # res_lp = solver_lp(model, test_S)
    # print(res_lp.is_success())
    # unstable_neurons_set = Search.BFS(test_S)
    # print(unstable_neurons_set)
    # print(len(unstable_neurons_set))