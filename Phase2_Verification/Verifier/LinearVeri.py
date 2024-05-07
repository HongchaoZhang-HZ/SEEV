from VeriUtil import *

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
    
def linearG():
    Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
    no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
    # If there is control input that can affect b, then return True meaning the sufficient verification is passed
    if not no_control_flag:
        return True