import numpy as np
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from pydrake.solvers import MathematicalProgram, Solve
from Modules.Function import RoA, LinearExp, solver_lp

def Lfx(S, x, Case):
    index_o = len(S.keys())-1
    W_o = np.array(W_o[index_o]).flatten()
    r_o = np.array(r_o[index_o])
    fx = Case.fx
    Lfx = W_o @ fx @ x
    return Lfx

def min_Lf(model, S, Case):
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    # Create an empty MathematicalProgram named prog (with no decision variables,
    # constraints or costs)
    prog = MathematicalProgram()
    # Add two decision variables x[0], x[1].
    dim = model.layer[0].in_features
    x = prog.NewContinuousVariables(dim, "x")
    prog = RoA(prog, x, model, S)
    # Add linear constraints
    W_B, r_B, W_o, r_o = LinearExp(S)
    print(W_B, r_B, W_o, r_o)
    
    # Output layer index
    index_o = len(S.keys())-1
    # Add linear constraints
    prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
    # Add cost function
    Lfx = Lfx(S, x, Case)
    LC = prog.AddLinearCost(Lfx)
    
    # Now solve the program.
    result = Solve(prog)
    print(f"Is solved successfully: {result.is_success()}")
    print(f"x optimal value: {result.GetSolution(x)}")
    print(f"optimal cost: {result.get_optimal_cost()}") 
