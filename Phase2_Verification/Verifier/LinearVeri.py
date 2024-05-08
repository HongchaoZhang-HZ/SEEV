from VeriUtil import *

class veri_seg_FG_wo_U(veri_seg_basic):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        
    def zero_Lg(self, x):
        # check if \frac{\partial b}{\partial x}G = 0, where 0 means the zero vector
        Lgb = self.W_out @ self.Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([self.dim, 1])).all()
        # If there is control input that can affect b, then return True meaning the sufficient verification is passed
        if not no_control_flag:
            return True
    
    def min_Lf(self, reverse_flat=False):
        # Input: X: Linear expression of the output of the ReLU NN
        # Output: X: Linear expression of the output of the ReLU NN
        # Create an empty MathematicalProgram named prog (with no decision variables,
        # constraints or costs)
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)

        # Add linear constraints
        prog.AddLinearConstraint(self.W_out @ x + self.r_out == 0)
        # Add cost function
        fx = self.Case.f_x(x)
        Lfb = (self.W_o[self.index_o] @ fx ).flatten()[0]
        if reverse_flat:
            LC = prog.AddCost(-Lfb)
        else:
            LC = prog.AddCost(Lfb)
        
        # Now solve the program.
        result = Solve(prog)
        return result.is_success(), result.GetSolution(x), result.get_optimal_cost()

class veri_seg_FG_with_interval_U(veri_seg_FG_wo_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.D = np.max(np.abs(self.Case.CTRLDIM))
        self.threshold_interval()
    
    def threshold_interval(self):
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(self.dim, "x")
        threshold = self.W_out @ self.Case.g_x(x) @ self.D
        self.thrshold = threshold
    
    def min_Lf_interval(self, reverse_flat=False):
        res_is_success, res_x, res_cost = self.min_Lf(reverse_flat)
        if res_cost + self.thrshold < 0:
            return False
        else:
            return True
        
class veri_seg_FG_with_con_U(veri_seg_FG_with_interval_U):
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)


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
    return result.is_success(), result.GetSolution(x), result.get_optimal_cost()
    
def linearG():
    Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
    no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
    # If there is control input that can affect b, then return True meaning the sufficient verification is passed
    if not no_control_flag:
        return True