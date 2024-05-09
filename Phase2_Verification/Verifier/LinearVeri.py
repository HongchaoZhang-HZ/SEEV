from tabnanny import check
from VeriUtil import *

class veri_seg_FG_wo_U(veri_seg_basic):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        
    def zero_Lg(self):
        # check if \frac{\partial b}{\partial x}G = 0, where 0 means the zero vector
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        Lgb = self.W_out @ self.Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([self.dim, 1])).all()
        # If there is control input that can affect b, then return True meaning the sufficient verification is passed
        if not no_control_flag:
            return True
    
    def min_Lf(self, reverse_flag=False):
        prog = MathematicalProgram()
        
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)

        # Add linear constraints
        prog.AddLinearConstraint(self.W_out @ x + self.r_out == 0)
        # Add cost function
        fx = self.Case.f_x(x)
        Lfb = (self.W_o[self.index_o] @ fx ).flatten()[0]
        if reverse_flag:
            LC = prog.AddCost(-Lfb)
        else:
            LC = prog.AddCost(Lfb)
        
        # Now solve the program.
        result = Solve(prog)
        return result.is_success(), result.GetSolution(x), result.get_optimal_cost()
    
    def verification(self, reverse_flag=False):
        check_Lg = self.zero_Lg()
        if check_Lg:
            return True, None
        res_is_success, res_x, res_cost = self.min_Lf(reverse_flag)
        if res_cost < 0:
            return False, res_x
        else:
            return True, None

class veri_seg_FG_with_interval_U(veri_seg_FG_wo_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.D = np.max(np.abs(self.Case.CTRLDOM), axis=1)
        self.threshold_interval()
    
    def threshold_interval(self):
        prog = MathematicalProgram()
        # Add two decision variables x[0], x[1].
        x = prog.NewContinuousVariables(self.dim, "x")
        threshold = self.W_out @ self.Case.g_x(x) @ self.D
        self.thrshold = threshold
    
    def min_Lf_interval(self, reverse_flag=False):
        res_is_success, res_x, res_cost = self.min_Lf(reverse_flag)
        if res_cost + self.thrshold < 0:
            return False, res_x
        else:
            return True, res_x
        
    def verification(self, reverse_flag=False):
        return self.min_Lf_interval(reverse_flag)
        
class veri_seg_FG_with_con_U(veri_seg_FG_with_interval_U):
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        
    def Farkas_lemma_BiLinear(self, reverse_flag=False, SMT_flag=False):
        pass
    
    def verification(self, reverse_flag=False):
        return self.Farkas_lemma_BiLinear(reverse_flag, SMT_flag=False)
    
class veri_hinge_FG_wo_U(veri_hinge_basic):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S_list):
        super().__init__(model, Case, S_list)
        self.W_out_list = [seg.W_out for seg in self.segs]
        self.r_out_list = [seg.r_out for seg in self.segs]
        
    def same_sign_Lg(self):
        # check if \frac{\partial b}{\partial x}G = 0, where 0 means the zero vector
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        Lgb_list = []
        for i in range(len(self.segs)):
            Lgb = self.W_out_list[i] @ self.Case.g_x(x)
            Lgb_list.append(np.sign(Lgb))
        # check if there is at least one row that has the same sign 
        same_sign_flag_list = []
        for i in range(self.Case.CTRLDIM):
            same_sign_flag = self.all_same(items=Lgb_list, idx=i)
            same_sign_flag_list.append(same_sign_flag)
        # if more than one, then return True meaning the sufficient verification is passed
        return any(same_sign_flag_list)
    
    def min_Lf_hinge(self, reverse_flag=False):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)
        # Add linear constraints
        for i in range(len(self.segs)):
            prog.AddLinearConstraint(self.W_out_list[i] @ x + self.r_out_list[i] == 0)
        # Add cost function
        fx = self.Case.f_x(x)
        Lfb_list = []
        for i in range(len(self.segs)):
            Lfb = (self.W_out_list[i] @ fx ).flatten()[0]
            if reverse_flag:
                LC = prog.AddCost(-Lfb)
            else:
                LC = prog.AddCost(Lfb)
            result = Solve(prog)
            # If all Lfb are positive, then return True meaning the sufficient verification is passed
            if result.get_optimal_cost() < 0:
                print('min_Lf_hinge = ', result.get_optimal_cost(), 'at point', result.GetSolution(x))
                return False, result.GetSolution(x)
            prog.RemoveCost(LC)
        return True, None
                
    def verification(self, reverse_flag=False):
        check_Lg = self.same_sign_Lg()
        if check_Lg:
            return True, None
        veri_flag, ce = self.min_Lf_hinge(reverse_flag)
        return veri_flag, ce

# def min_Lf(model, S, Case, reverse_flag=False):
#     # Input: X: Linear expression of the output of the ReLU NN
#     # Output: X: Linear expression of the output of the ReLU NN
#     # Create an empty MathematicalProgram named prog (with no decision variables,
#     # constraints or costs)
#     prog = MathematicalProgram()
#     # Add two decision variables x[0], x[1].
#     dim = Case.DIM
#     x = prog.NewContinuousVariables(dim, "x")
#     prog = RoA(prog, x, model, S)
#     # Add linear constraints
#     W_B, r_B, W_o, r_o = LinearExp(model, S)
    
#     # Output layer index
#     index_o = len(S.keys())-1
    
#     # Add linear constraints
#     prog.AddLinearConstraint(np.array(W_o[index_o]).flatten() @ x + np.array(r_o[index_o]) == 0)
#     # Add cost function
#     fx = Case.f_x(x)
#     Lfb = (W_o[index_o] @ fx ).flatten()[0]
#     if reverse_flag:
#         LC = prog.AddCost(-Lfb)
#     else:
#         LC = prog.AddCost(Lfb)
    
#     # Now solve the program.
#     result = Solve(prog)
#     return result.is_success(), result.GetSolution(x), result.get_optimal_cost()
    
# def linearG():
#     Lgb = np.array(W_o[index_o]).flatten() @ Case.g_x(x)
#     no_control_flag = np.equal(Lgb, np.zeros([Case.CTRLDIM, 1])).all()
#     # If there is control input that can affect b, then return True meaning the sufficient verification is passed
#     if not no_control_flag:
#         return True