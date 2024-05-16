import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
import Scripts.PARA as PARA
from Modules.Function import *
from Modules.visualization_module import *

class SearchInit:
    def __init__(self, model, case=None) -> None:
        self.zero_tol = PARA.zero_tol
        self.model = model
        self.dim = next(model.children())[0].in_features
        self.case = case
        self.NStatus = NetworkStatus(model)
        # if case is not None:
        #     self.safe_regions = self.case.safe_regions
        #     self.unsafe_regions = self.case.unsafe_regions
    
    def sample_in_region(self, region, len_sample):
        grid_sample = [torch.linspace(region[i][0], region[i][1], int(len_sample[i])) for i in range(len(region))]
        return torch.meshgrid(grid_sample)
    
    def connect_region(self, region, len_sample):
        mesh = self.sample_in_region(region, len_sample)
        flatten = [torch.flatten(mesh[i]) for i in range(len(mesh))]
        return torch.stack(flatten, 1)
    
    def identification_lp(self, S):
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(self.dim, "x")
        # Add linear constraints
        W_B, r_B, W_o, r_o = LinearExp(self.model, S)
        prog = RoA(prog, x, self.model, S=None, W_B=W_B, r_B=r_B, SSpace=self.case.SSpace)
        
        index_o = len(S.keys())-1
        prog.AddLinearEqualityConstraint(np.array(W_o[index_o]), -np.array(r_o[index_o]), x)
        # prog.AddCost(np.array(W_o[index_o]@x + r_o[index_o]))
        # print(W_o[index_o], r_o[index_o])
        
        result = Solve(prog)
        return result
    
    def get_zero_S(self, p_safe, p_unsafe, iter_lim = 100):
        # conduct binary search to find the zero point
        # p_safe and p_unsafe are two points with different signs of self.model.forward
        # return the zero point where self.model.forward is zero
        # the zero point can be found by solving linear programming
        # flag True/False indicates whether the zero point is found by the LP
        # If flag is False, check the sign of LP result to see if it is positive or negative
        # if negative check the left side of the interval, otherwise check the right side
        flag = False
        S = None
        
        # self.NStatus.get_netstatus_from_input(p_safe)
        # # print(p_safe, mid_point, p_unsafe)
        # S = self.NStatus.network_status_values
        # id_flag = solver_lp(self.model, S).is_success()
        # if id_flag:
        #     flag = True
        #     return flag, S
            
        # self.NStatus.get_netstatus_from_input(p_unsafe)
        # # print(p_safe, mid_point, p_unsafe)
        # S = self.NStatus.network_status_values
        # id_flag = solver_lp(self.model, S).is_success()
        # if id_flag:
        #     flag = True
        #     return flag, S
        
        for iter in range(iter_lim):
            mid_point = (p_safe + p_unsafe) / 2
            self.NStatus.get_netstatus_from_input(mid_point)
            # print(p_safe, mid_point, p_unsafe)
            S = self.NStatus.network_status_values
            id_flag = solver_lp(self.model, S, SSpace=self.case.SSpace).is_success()
            if id_flag:
                flag = True
                return flag, S
            # elif self.model.forward(mid_point) > 0:
            elif torch.sign(self.model.forward(mid_point)) * torch.sign(self.model.forward(p_safe)) < 0:
                p_unsafe = mid_point
            else:
                p_safe = mid_point
        return flag, S
    
    def initialization(self, input_safe:torch.tensor=None, input_unsafe:torch.tensor=None, m = 1000):
        # define default flag and S_init_Set
        flag, S_init_Set = False, {}
        # if we specify input_safe and input_unsafe, we use the input directly
        if input_safe is not None and input_unsafe is not None:
            m = input_safe.shape[0]
            safe_list_length = input_safe.shape[1]
            unsafe_list_length = input_unsafe.shape[1]
            x_safe = input_safe
            x_unsafe = input_unsafe
        elif input_safe is None:
            safe_list_length = len(self.safe_regions)
        elif input_unsafe is None:
            unsafe_list_length = len(self.unsafe_regions)
        else:
            safe_list_length = len(self.safe_regions)
            unsafe_list_length = len(self.unsafe_regions)
        # iterate through all safe and unsafe regions to find all intial activation sets
        for i in range(safe_list_length):
            for j in range(unsafe_list_length):
                if input_safe is None and input_unsafe is None:
                    x_safe = self.sample_in_region(self.safe_regions[i], [m, 1])
                    x_unsafe = self.sample_in_region(self.unsafe_regions[j], [m, 1])
                else:
                    x_safe = input_safe[i]
                    x_unsafe = input_unsafe[j]
                if not flag:
                    for k in range(m):
                        flag, S = self.get_zero_S(x_safe[0], x_unsafe[0]) # only check the first unsafe point
                        if flag:
                            S_init_Set[j] = S
        if not flag:
            raise("Initialization failed")
        return S_init_Set


if __name__ == "__main__":
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase1_Scalability/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    # case = PARA.CASES[0]
    Search = SearchInit(model)
    # (0.5, 1.5), (0, -1)
    S_init_Set = Search.initialization(torch.tensor([[[0.5, 1.7]]]), torch.tensor([[[-1, -1.6]]]))
    # S_init_Set = Search.initialization(input_safe=torch.tensor([[[0.5, 1.5]]]), input_unsafe=torch.tensor([[[-1, 0]]]))
    print(S_init_Set)
