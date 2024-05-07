import numpy as np
import torch
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from pydrake.solvers import MathematicalProgram, Solve
from Modules.Function import RoA, LinearExp, solver_lp

class verifier_basic():
    def __init__(self, model, Case):
        self.model = model
        self.Case = Case
        self.dim = Case.DIM
        self.x = None
        self.W_B = None
        self.r_B = None
        self.W_o = None
        self.r_o = None
        self.index_o = None
        self.Lfb = None
        self.Lgb = None
        self.no_control_flag = None
        self.result = None
        self.prog
