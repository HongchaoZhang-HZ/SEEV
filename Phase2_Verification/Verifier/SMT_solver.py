import dreal as dr
import numpy as np
from Verifier.VeriUtil import *
from Modules.Function import *

def RoA_dreal(model, S):
    W_B, r_B, W_o, r_o = LinearExp(model, S)
    W_out = np.array(W_o[len(S.keys())-1])
    r_out = np.array(r_o[len(S.keys())-1])
    
    final_layer_idx = len(W_B.keys())
    assert len(W_B) == len(r_B), "W_B and r_B must have the same amount of layers"
    for keys, layer_info in W_B.items():
        if keys == 0:
            cons_W = layer_info
            cons_r = r_B[keys]
        elif keys == final_layer_idx:
            break
        else:
            cons_W = np.vstack([cons_W, layer_info])
            cons_r = np.hstack([cons_r, r_B[keys]])
    return cons_W, cons_r, W_out, r_out

def XSi_b0_dreal(model, x_vars, S_list, n, output_flag=False):
    constraints = []
    if isinstance(S_list, dict):
        S_list = [S_list]
    W_out_list = []
    r_out_list = []
    for S in S_list:
        cons_W, cons_r, W_out, r_out = RoA_dreal(model, S)
        W_out_list.append(W_out)
        r_out_list.append(r_out)
        for i in range(cons_W.shape[0]):
            linear_expr = sum(cons_W[i, j] * x_vars[j] for j in range(n)) + cons_r[i]
            constraints.append(linear_expr >= 0)
        bx0 = sum(W_out[0, j] * x_vars[j] for j in range(n)) + r_out[0]
        constraints.append(bx0 == 0)
    if output_flag:
        return constraints, W_out_list, r_out_list
    return constraints

def check_negative_minfx(model, S, case, reverse_flag=False):
    # Extract necessary information for the constraints
    cons_W, cons_r, W_out, r_out = RoA_dreal(model, S)
    n = case.DIM  # Dimension of the problem

    # Define scalar variables to represent components of a vector
    x_vars = [dr.Variable(f"x{i}") for i in range(n)]

    # Setup constraints using dReal, manually converting from NumPy operations
    constraints = XSi_b0_dreal(model, x_vars, S, n)
    
    fx = case.f_x_dreal(x_vars)
    Lfb = sum(W_out[0, j] * fx[j] for j in range(n))
    
    # Function f to minimize and constraint to check its negativity
    if reverse_flag:
        constraints.append(Lfb > 0)
    else:
        constraints.append(Lfb < 0)  # Add constraint that f is negative

    # Define the configuration for the solver
    config = dr.Config()
    config.use_polytope_in_forall = True
    config.precision = 1e-6

    # Check for the existence of a solution
    formula = dr.And(*constraints)
    result = dr.CheckSatisfiability(formula, config)

    # Analyze the result
    if result:
        print("SAT: There is a solution where the function is negative.")
        ce = np.zeros(n)
        for i, var in enumerate(x_vars):
            interval = result[var]
            midpoint = (interval.lb() + interval.ub()) / 2
            ce[i] = midpoint
            print(f"Counter example x{i+1} = {midpoint} (interval: {interval})")
        return False, ce
    else:
        print("UNSAT: No solution exists where the function is negative.")
        return True, None

def check_negative_minfx_gx0(model, S, case, reverse_flag=False):
    # Extract necessary information for the constraints
    cons_W, cons_r, W_out, r_out = RoA_dreal(model, S)
    n = case.DIM  # Dimension of the problem

    # Define scalar variables to represent components of a vector
    x_vars = [dr.Variable(f"x{i}") for i in range(n)]

    # Setup constraints using dReal, manually converting from NumPy operations
    constraints = XSi_b0_dreal(model, x_vars, S, n)
    
    # constraint dbdxg(x)=0
    gx = case.g_x_dreal(x_vars)
    Lgb = sum(W_out[0, j] * gx[j] for j in range(n))
    constraints.append(Lgb == 0)
    
    fx = case.f_x_dreal(x_vars)
    Lfb = sum(W_out[0, j] * fx[j] for j in range(n))
    
    # Function f to minimize and constraint to check its negativity
    if reverse_flag:
        constraints.append(Lfb > 0)
    else:
        constraints.append(Lfb < 0)  # Add constraint that f is negative

    # Define the configuration for the solver
    config = dr.Config()
    config.use_polytope_in_forall = True
    config.precision = 1e-6

    # Check for the existence of a solution
    formula = dr.And(*constraints)
    result = dr.CheckSatisfiability(formula, config)

    # Analyze the result
    if result:
        print("SAT: There is a solution where the function is negative.")
        ce = np.zeros(n)
        for i, var in enumerate(x_vars):
            interval = result[var]
            midpoint = (interval.lb() + interval.ub()) / 2
            ce[i] = midpoint
            print(f"Counter example x{i+1} = {midpoint} (interval: {interval})")
        return False, ce
    else:
        print("UNSAT: No solution exists where the function is negative.")
        return True, None

def farkas_lemma_seg(model, S, case, A, c, reverse_flag=False):
    # Extract necessary information for the constraints
    # cons_W, cons_r, W_out, r_out = RoA_dreal(model, S)
    n = case.DIM  # Dimension of the problem
    ny = 1 + len(A)
    # Define scalar variables to represent components of a vector
    x_vars = [dr.Variable(f"x{i}") for i in range(n)]
    y_vars = [dr.Variable(f"y{i}") for i in range(ny)]

    # Setup constraints using dReal, manually converting from NumPy operations
    constraints, W_out_list, r_out_list = XSi_b0_dreal(model, x_vars, [S], n, output_flag=True)
    
    # construct Theta
    Theta = []
    gx = case.g_x_dreal(x_vars)
    for i in range(len([S])):
        Lgb = sum(W_out_list[i][0, j] * gx[j] for j in range(n))
        Theta.append(Lgb)
    for i in range(len(A)):
        Theta.append(A[i])
    # yT theta = 0
    constraints.append(sum(y_vars[i] * Theta[i] for i in range(ny)) == 0)
    
    # construct Lambda
    Lambda = []
    fx = case.f_x_dreal(x_vars)
    for i in range(len([S])):
        if reverse_flag:
            Lfb = -sum(W_out_list[i][0, j] * fx[j] for j in range(n))
        else:
            Lfb = sum(W_out_list[i][0, j] * fx[j] for j in range(n))
        Lambda.append(Lfb)
    for i in range(len(c)):
        Lambda.append(c[i])
    # yT Lambda < 0
    constraints.append(sum(y_vars[i] * Lambda[i] for i in range(ny)) < 0)
    
    # y >= 0
    constraints.append(y_vars >= 0)

    # Define the configuration for the solver
    config = dr.Config()
    config.use_polytope_in_forall = True
    config.precision = 1e-6

    # Check for the existence of a solution
    formula = dr.And(*constraints)
    result = dr.CheckSatisfiability(formula, config)

    # Analyze the result
    if result:
        print("SAT: There is a solution where the function is negative.")
        ce = np.zeros(n)
        for i, var in enumerate(x_vars):
            interval = result[var]
            midpoint = (interval.lb() + interval.ub()) / 2
            ce[i] = midpoint
            print(f"Counter example x{i+1} = {midpoint} (interval: {interval})")
        return False, ce
    else:
        print("UNSAT: No solution exists where the function is negative.")
        return True, None

def farkas_lemma_hinge(model, S_list, case, A, c, reverse_flag=False):
    # Extract necessary information for the constraints
    # cons_W, cons_r, W_out, r_out = RoA_dreal(model, S_list)
    n = case.DIM  # Dimension of the problem
    ny = len(S_list) + len(A)
    # Define scalar variables to represent components of a vector
    x_vars = [dr.Variable(f"x{i}") for i in range(n)]
    y_vars = [dr.Variable(f"y{i}") for i in range(ny)]

    # Setup constraints using dReal, manually converting from NumPy operations
    constraints, W_out_list, r_out_list = XSi_b0_dreal(model, x_vars, S_list, n, output_flag=True)
    
    # construct Theta
    Theta = []
    gx = case.g_x_dreal(x_vars)
    for i in range(len(S_list)):
        Lgb = sum(W_out_list[i][0, j] * gx[j] for j in range(n))
        Theta.append(Lgb)
    for i in range(len(A)):
        Theta.append(A[i])
    # yT theta = 0
    constraints.append(sum(y_vars[i] * Theta[i] for i in range(ny)) == 0)
    
    # construct Lambda
    Lambda = []
    fx = case.f_x_dreal(x_vars)
    for i in range(len(S_list)):
        if reverse_flag:
            Lfb = -sum(W_out_list[i][0, j] * fx[j] for j in range(n))
        else:
            Lfb = sum(W_out_list[i][0, j] * fx[j] for j in range(n))
        Lambda.append(Lfb)
    for i in range(len(c)):
        Lambda.append(c[i])
    # yT Lambda < 0
    constraints.append(sum(y_vars[i] * Lambda[i] for i in range(ny)) < 0)
    
    # y >= 0
    constraints.append(y_vars[i] >= 0 for i in range(ny))

    # Define the configuration for the solver
    config = dr.Config()
    config.use_polytope_in_forall = True
    config.precision = 1e-6

    # Check for the existence of a solution
    formula = dr.And(*constraints)
    result = dr.CheckSatisfiability(formula, config)

    # Analyze the result
    if result:
        print("SAT: There is a solution where the function is negative.")
        ce = np.zeros(n)
        for i, var in enumerate(x_vars):
            interval = result[var]
            midpoint = (interval.lb() + interval.ub()) / 2
            ce[i] = midpoint
            print(f"Counter example x{i+1} = {midpoint} (interval: {interval})")
        return False, ce
    else:
        print("UNSAT: No solution exists where the function is negative.")
        return True, None

# Run the function to check for negative optimum
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
    
# check_negative_minfx(model, Search_prog.S_init[0], case, reverse_flag=True)
# check_negative_minfx_gx0(model, Search_prog.S_init[0], case, reverse_flag=True)
