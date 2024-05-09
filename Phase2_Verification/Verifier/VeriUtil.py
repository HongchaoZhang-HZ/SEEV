from tkinter import N
import numpy as np
import torch
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from pydrake.solvers import MathematicalProgram, Solve
from Modules.Function import RoA, LinearExp, solver_lp

class veri_seg_basic():
    def __init__(self, model, Case, S):
        self.model = model
        self.S = S
        self.Case = Case
        self.dim = Case.DIM
        self._update_linear_exp(self.S)
        self.NChx = self.Case.NChx
        self.feasibility_only = False
        
    def _update_linear_exp(self, S):
        self.W_B, self.r_B, self.W_o, self.r_o = LinearExp(self.model, S)
        self.index_o = len(S.keys())-1
        self.W_out = np.array(self.W_o[self.index_o]).flatten()
        self.r_out = np.array(self.r_o[self.index_o])
        
    def XS(self, prog, x, S=None):
        if S == None:
            S = self.S
        prog = RoA(prog, x, self.model, S)
        return prog
    
    def correctness_LP(self, pos_safe_flag=True):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)

        # Add linear constraints
        prog.AddLinearConstraint(self.W_out @ x + self.r_out == 0)
        # Add cost function
        hx = self.Case.h_x(x)
        if pos_safe_flag:
            LC = prog.AddCost(hx)
        else:
            LC = prog.AddCost(-hx)
        
        # Now solve the program.
        result = Solve(prog)
        return result.is_success(), result.GetSolution(x), result.get_optimal_cost()
    
    def correctness_SMT(self, pos_safe_flag=True):
        pass
    
    def veri_correctness(self, pos_safe_flag=True):
        if not self.NChx:
            veri_flag, ce, cost = self.correctness_LP(pos_safe_flag)
        else:
            veri_flag, ce, cost = self.correctness_SMT(pos_safe_flag)
        if cost >= 0:
            return True, None
        else:
            print('correctness ce with h(x) = ', cost, 'at point', ce)
            return False, ce
        
class veri_hinge_basic():
    def __init__(self, model, Case, S_list):
        self.model = model
        self.Case = Case
        self.S_list = S_list
        self.dim = Case.DIM
        self._init_segs()
        
    def _init_segs(self):
        self.segs = []
        for S in self.S_list:
            self.segs.append(veri_seg_basic(self.model, self.Case, S))

    def XSi(self, prog, x, i):
        return self.segs[i].XS(prog, x)
    
    def XS(self, prog, x):
        for i in range(len(self.segs)):
            prog = self.XSi(prog, x, i)
        return prog
    
    def all_same(self, items:list, idx=None):
        if idx != None:
            return all(item[idx] == items[0][idx] for item in items)
        else:
            return all(item == items[0] for item in items)

class verifier_basic():
    def __init__(self, model, Case):
        self.model = model
        self.Case = Case
        self.dim = Case.DIM

        self.is_gx_linear = Case.is_gx_linear
        self.is_fx_linear = Case.is_fx_linear
        self.is_u_cons = Case.is_u_cons
        self.is_u_cons_interval = Case.is_u_cons_interval
        
