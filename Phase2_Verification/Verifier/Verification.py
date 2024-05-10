import sys, os
import time
from pytest import fail
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from VeriUtil import *
from LinearVeri import *
from NLVeri import *

from Modules.NNet import NeuralNetwork as NNet
from Cases.Darboux import Darboux
from Cases.ObsAvoid import ObsAvoid
from Cases.LinearSatellite import LinearSat
from Scripts.Search import Search

class Verifier(verifier_basic):
    def __init__(self, model, Case, 
                 unstable_neurons_set, 
                 pair_wise_hinge, 
                 ho_hinge, VNBC_flag=False):
        super().__init__(model, Case)
        self.unstable_neurons_set = unstable_neurons_set
        self.pair_wise_hinge = pair_wise_hinge
        self.ho_hinge = ho_hinge
        self.diagnose()
        self.feasibility_only = False
        
        # T - Barrier Certificate Verification, 
        # F - Control Barrier Function Verification
        self.VNBC_flag = VNBC_flag 
        
    def diagnose(self):
        if self.is_gx_linear:
            type_g = 'G'
        else:
            type_g = 'g'
        if self.is_fx_linear:
            type_f = 'F'
        else:
            type_f = 'f'
        if self.is_u_cons:
            type_u = 'U'
            if self.is_u_cons_interval:
                type_u += 'i'
            else:
                type_u += 'c'
        else:
            type_u = 'N'
        self.type = type_f + type_g + type_u
        
    def seg_verifier(self, S):
        if self.type == 'FGN':
            return veri_seg_FG_wo_U(self.model, self.Case, S)
        elif self.type == 'fGN':
            return veri_seg_fG_wo_U(self.model, self.Case, S)
        elif self.type == 'FgN':
            return veri_seg_Fg_wo_U(self.model, self.Case, S)
        elif self.type == 'fgN':
            return veri_seg_Nfg_wo_U(self.model, self.Case, S)
        elif self.type == 'FGUi':
            return veri_seg_FG_with_interval_U(self.model, self.Case, S)
        elif self.type == 'fGUi':
            return veri_seg_fG_with_interval_U(self.model, self.Case, S)
        elif self.type == 'FgUi' or self.type == 'fgUi':
            return veri_seg_Nfg_with_interval_U(self.model, self.Case, S)
        elif self.type == 'FGUc':
            return veri_seg_FG_with_con_U(self.model, self.Case, S)
        elif self.type == 'FgUc' or self.type == 'fGUc' or self.type == 'fgUc':
            return veri_seg_Nfg_with_con_U(self.model, self.Case, S)
        
    def hinge_verifier(self, S):
        if self.type == 'FGN':
            return veri_hinge_FG_wo_U(self.model, self.Case, S)
        elif self.type == 'fGN':
            return veri_hinge_fG_wo_U(self.model, self.Case, S)
        elif self.type == 'FgN':
            return veri_hinge_Fg_wo_U(self.model, self.Case, S)
        elif self.type == 'fgN':
            return veri_hinge_Nfg_wo_U(self.model, self.Case, S)
        else:
            return veri_hinge_Nfg_cons_U(self.model, self.Case, S)
        
    def seg_verification(self, unstable_neurons_set, reverse_flag=False):
        for S in unstable_neurons_set:
            seg_verifier = self.seg_verifier(S)
            veri_flag, ce = seg_verifier.verification(reverse_flag, self.feasibility_only)
            if not veri_flag:
                print('Verification failed!')
                print('Segment counter example', ce)
                return False, ce
        return True, None
    
    def hinge_verification(self, hinge_set, reverse_flag=False):
        for hinge in hinge_set:
            hinge_verifier = self.hinge_verifier(hinge)
            veri_flag, ce = hinge_verifier.verification(reverse_flag)
            if not veri_flag:
                print('Verification failed!')
                print('Hinge counter example', ce)
                return False, ce
        return True, None
    
    def Verification(self, reverse_flag=False):
        veri_flag_seg, ce_seg = self.seg_verification(self.unstable_neurons_set, reverse_flag)
        if not veri_flag_seg:
            return False, ce_seg
        veri_flag_hinge, ce_hinge = self.hinge_verification(self.pair_wise_hinge, reverse_flag)
        if not veri_flag_hinge:
            return False, ce_hinge
        veri_flag_ho, ce_ho = self.hinge_verification(self.ho_hinge, reverse_flag)
        if not veri_flag_ho:
            return False, ce_ho
        print('Verification passed!')
        return True, None

if __name__ == "__main__":
    
    # # BC Verification
    # case = Darboux()
    # architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    # model = NNet(architecture)
    # trained_state_dict = torch.load("./Phase2_Verification/models/darboux_1_32.pt")
    # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    # model.load_state_dict(trained_state_dict, strict=True)
    
    # time_start = time.time()
    # Search_prog = Search(model)
    # Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0]]]))
    # unstable_neurons_set, pairwise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    # ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pairwise_hinge) 
    # search_time = time.time() - time_start
    
    # Verifier = Verifier(model, case, unstable_neurons_set, pairwise_hinge, ho_hinge)
    # veri_flag, ce = Verifier.Verification(reverse_flag=True)
    # verification_time = time.time() - time_start - search_time
    # print('Search time:', search_time)
    # print('Verification time:', verification_time)
    
    # # CBF Verification
    # case = ObsAvoid()
    # architecture = [('linear', 3), ('relu', 64), ('relu', 64), ('linear', 1)]
    # model = NNet(architecture)
    # trained_state_dict = torch.load("Phase1_Scalability/models/obs_2_64.pt")
    # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    # model.load_state_dict(trained_state_dict, strict=True)
    
    # time_start = time.time()
    # Search_prog = Search(model)
    # spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    # uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    # Search_prog.Specify_point(spt, uspt)
    # unstable_neurons_set, pair_wise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    # ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pair_wise_hinge)
    # search_time = time.time() - time_start
    
    # Verifier = Verifier(model, case, unstable_neurons_set, pair_wise_hinge, ho_hinge)
    # veri_flag, ce = Verifier.Verification(reverse_flag=True)
    # verification_time = time.time() - time_start - search_time
    # print('Search time:', search_time)
    # print('Verification time:', verification_time)
    
    # CBF Verification
    case = LinearSat()
    architecture = [('linear', 6), ('relu', 16), ('relu', 16), ('linear', 16), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase2_Verification/models/satellitev1_2_16.pt")
    renamed_state_dict = model.wrapper_load_state_dict(trained_state_dict)
    # Load the renamed state dict into the model
    model.load_state_dict(renamed_state_dict, strict=True)
    model.merge_last_n_layers(2)
    
    time_start = time.time()
    Search_prog = Search(model)
    spt = torch.tensor([[[-1.2, -1.5, 1.1, 0.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
    Search_prog.Specify_point(spt, uspt)
    unstable_neurons_set, pair_wise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    ho_hinge = Search_prog.hinge_search(unstable_neurons_set, pair_wise_hinge)
    search_time = time.time() - time_start
    
    Verifier = Verifier(model, case, unstable_neurons_set, pair_wise_hinge, ho_hinge)
    veri_flag, ce = Verifier.Verification(reverse_flag=True)
    verification_time = time.time() - time_start - search_time
    print('Search time:', search_time)
    print('Verification time:', verification_time)

# def check_Lg_wo_U(model, S, Case):
#     prog = MathematicalProgram()
#     # Add two decision variables x[0], x[1].
#     dim = Case.DIM
#     x = prog.NewContinuousVariables(dim, "x")
#     prog = RoA(prog, x, model, S)
#     # Add linear constraints
#     W_B, r_B, W_o, r_o = LinearExp(model, S)
    
#     # Output layer index
#     index_o = len(S.keys())-1
    
#     # For cases with linear G, then we can directly check if Lgb == 0
#     if Case.linear_gx:
#         Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
#         no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
#         # If there is control input that can affect b, then return True meaning the sufficient verification is passed
#         if not no_control_flag:
#             return True
#     else:
#         # Add linear constraints
#         prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
#         # TODO: check if nonlinear case would have x in side Lgb
#         Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
#         no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
#         prog.AddConstraint(no_control_flag)
#         result = Solve(prog)
#     return result.is_success()

# def suff_speed_up(model, S, Case, U_cons_flag=False):
#     if not U_cons_flag:
#         return check_Lg_wo_U(model, S, Case)
#     # TODO: add the sufficient verification with control input constraints
    

# def BC_verification(model, Case, reverse_flat=True, immediate_return=False):
#     Search_prog = Search(model)
#     # (0.5, 1.5), (0, -1)
#     Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0.1]]]))

#     unstable_neurons_set, pairwise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    
#     CE = []
#     # Verify each boundary segments
#     for item in unstable_neurons_set:
#         res_flag, res_x, res_cost = min_Lf(model, item, Case, reverse_flat=reverse_flat)
#         if res_cost < 0:
#             CE.append((item, res_x))
#             if immediate_return:
#                 print("Verification failed!")
#                 return False, CE
            
#     # TODO: Add the verification of the vertices
#     # Verify each vertices
    
#     print("Verification passed!")
#     return True, CE

# def CBF_verification(model, Case, reverse_flat=True, U_cons_flag=False, immediate_return=False):
#     Search_prog = Search(model)
#     # (0.5, 1.5), (0, -1)
#     spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
#     uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
#     Search_prog.Specify_point(spt, uspt)
#     unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    
#     CE = []
#     # Verify each boundary segments
#     for item in unstable_neurons_set:
#         # suff_flag means the sufficient verification is passed
#         suff_flag = suff_speed_up(model, item, Case, U_cons_flag=U_cons_flag)
#         # if pass, then skip the following verification
#         if suff_flag:
#             continue
#         res_flag, res_x, res_cost = min_Lf(model, item, Case, reverse_flat=reverse_flat)
#         if res_cost < 0:
#             CE.append((item, res_x))
#             if immediate_return:
#                 print("Verification failed!")
#                 return False, CE
            
#     # TODO: Add the verification of the vertices
#     # Verify each vertices
    
#     print("Verification passed!")
#     return True, CE

# if __name__ == "__main__":
    
    
#     # BC Verification
#     case = Darboux()
#     architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
#     model = NNet(architecture)
#     trained_state_dict = torch.load("./Phase2_Verification/models/darboux_1_32.pt")
#     trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
#     model.load_state_dict(trained_state_dict, strict=True)
#     VeriRes, CE = BC_verification(model, case, reverse_flat=True)
    
#     # # CBF Verification
#     # case = ObsAvoid()
#     # architecture = [('linear', 3), ('relu', 64), ('linear', 1)]
#     # model = NNet(architecture)
#     # trained_state_dict = torch.load("Phase1_Scalability/models/obs_1_64.pt")
#     # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
#     # model.load_state_dict(trained_state_dict, strict=True)
#     # VeriRes, CE = CBF_verification(model, case, reverse_flat=True)
    
#     unstable_neurons_set, pair_wise_hinge = Search.BFS(Search.S_init[0])
#     ho_hinge = Search.hinge_search(unstable_neurons_set, pair_wise_hinge)