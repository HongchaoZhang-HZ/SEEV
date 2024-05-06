from audioop import reverse
import re
import numpy as np
import torch
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from pydrake.solvers import MathematicalProgram, Solve
from Modules.Function import RoA, LinearExp, solver_lp

from Modules.NNet import NeuralNetwork as NNet
from Cases.Darboux import Darboux
from Cases.ObsAvoid import ObsAvoid
from Scripts.Search import Search


def min_Lf(model, S, Case, reverse_flat=False):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = Case.DIM
    x = prog.NewContinuousVariables(dim, "x")
    prog = RoA(prog, x, model, S)
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    
    # Output layer index
    index_o = len(S.keys())-1
    
    # Add linear constraints
    prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
    # Add cost function
    fx = Case.f_x(x)
    Lfb = (W_o[index_o] @ fx ).flatten()[0]
    if reverse_flat:
        LC = prog.AddCost(-Lfb)
    else:
        LC = prog.AddCost(Lfb)
    
    # Now solve the program.
    result = Solve(prog)
    # print(f"Is solved successfully: {result.is_success()}")
    # print(f"x optimal value: {result.GetSolution(x)}")
    # print(f"optimal cost: {result.get_optimal_cost()}") 
    return result.is_success(), result.GetSolution(x), result.get_optimal_cost()
    
def check_Lg_wo_U(model, S, Case):
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = Case.DIM
    x = prog.NewContinuousVariables(dim, "x")
    prog = RoA(prog, x, model, S)
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    
    # Output layer index
    index_o = len(S.keys())-1
    
    # For cases with linear G, then we can directly check if Lgb == 0
    if Case.linear_gx:
        Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
        # If there is control input that can affect b, then return True meaning the sufficient verification is passed
        if not no_control_flag:
            return True
    else:
        # Add linear constraints
        prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
        # TODO: check if nonlinear case would have x in side Lgb
        Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
        prog.AddConstraint(no_control_flag)
        result = Solve(prog)
    return result.is_success()

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
    
