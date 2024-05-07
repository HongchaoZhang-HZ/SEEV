import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from Modules.NNet import NeuralNetwork as NNet
from Cases.Darboux import Darboux
from Cases.ObsAvoid import ObsAvoid
from Scripts.Search import Search

from LinearVeri import *

def suff_speed_up(model, S, Case, U_cons_flag=False):
    if not U_cons_flag:
        return check_Lg_wo_U(model, S, Case)
    # TODO: add the sufficient verification with control input constraints
    

def BC_verification(model, Case, reverse_flat=True, immediate_return=False):
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    Search_prog.Specify_point(torch.tensor([[[0.5, 1.5]]]), torch.tensor([[[-1, 0.1]]]))

    unstable_neurons_set, pairwise_hinge = Search_prog.BFS(Search_prog.S_init[0])
    
    CE = []
    # Verify each boundary segments
    for item in unstable_neurons_set:
        res_flag, res_x, res_cost = min_Lf(model, item, Case, reverse_flat=reverse_flat)
        if res_cost < 0:
            CE.append((item, res_x))
            if immediate_return:
                print("Verification failed!")
                return False, CE
            
    # TODO: Add the verification of the vertices
    # Verify each vertices
    
    print("Verification passed!")
    return True, CE

def CBF_verification(model, Case, reverse_flat=True, U_cons_flag=False, immediate_return=False):
    Search_prog = Search(model)
    # (0.5, 1.5), (0, -1)
    spt = torch.tensor([[[-1.0, 0.0, 0.0]]])
    uspt = torch.tensor([[[0.0, 0.0, 0.0]]])
    Search_prog.Specify_point(spt, uspt)
    unstable_neurons_set = Search_prog.BFS(Search_prog.S_init[0])
    
    CE = []
    # Verify each boundary segments
    for item in unstable_neurons_set:
        # suff_flag means the sufficient verification is passed
        suff_flag = suff_speed_up(model, item, Case, U_cons_flag=U_cons_flag)
        # if pass, then skip the following verification
        if suff_flag:
            continue
        res_flag, res_x, res_cost = min_Lf(model, item, Case, reverse_flat=reverse_flat)
        if res_cost < 0:
            CE.append((item, res_x))
            if immediate_return:
                print("Verification failed!")
                return False, CE
            
    # TODO: Add the verification of the vertices
    # Verify each vertices
    
    print("Verification passed!")
    return True, CE

if __name__ == "__main__":
    
    
    # BC Verification
    case = Darboux()
    architecture = [('linear', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    trained_state_dict = torch.load("./Phase2_Verification/models/darboux_1_32.pt")
    trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    model.load_state_dict(trained_state_dict, strict=True)
    VeriRes, CE = BC_verification(model, case, reverse_flat=True)
    
    # # CBF Verification
    # case = ObsAvoid()
    # architecture = [('linear', 3), ('relu', 64), ('linear', 1)]
    # model = NNet(architecture)
    # trained_state_dict = torch.load("Phase1_Scalability/models/obs_1_64.pt")
    # trained_state_dict = {f"layers.{key}": value for key, value in trained_state_dict.items()}
    # model.load_state_dict(trained_state_dict, strict=True)
    # VeriRes, CE = CBF_verification(model, case, reverse_flat=True)
    
