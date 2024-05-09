from math import e
import re

from torch import zero_
from LinearVeri import *

class veri_seg_Fg_wo_U(veri_seg_FG_wo_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.SMT_flag = False
        
    def zero_NLg(self):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)
        # check if \frac{\partial b}{\partial x}G = 0, where 0 means the zero vector
        Lgb = self.W_out @ self.Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([self.dim, 1])).all()
        prog.AddLinearConstraint(no_control_flag)
        # If there is control input that can affect b, then return True meaning the sufficient verification is passed
        res = Solve(prog)
        IsAllZero = res.get_optimal_cost()
        if not IsAllZero:
            return True
    
    def min_LFNLg(self, reverse_flag=False):
        prog = MathematicalProgram()
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)
        # check if \frac{\partial b}{\partial x}G = 0, where 0 means the zero vector
        Lgb = self.W_out @ self.Case.g_x(x)
        no_control_flag = np.equal(Lgb, np.zeros([self.dim, 1])).all()
        prog.AddLinearConstraint(no_control_flag)
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
        Lf_u0_flag, Lf_u0_res_x, Lf_u0_res_cost = self.min_Lf(reverse_flag)
        if Lf_u0_res_cost >= 0:
            return True, None
        # quick check if Lgb is zero
        if not self.SMT_flag:
            check_Lg = self.zero_Lg()
            if check_Lg:
                return True, None
        res_is_success, res_x, res_cost = self.min_LFNLg(reverse_flag)
        if res_cost < 0:
            return False, res_x
        else:
            return True, None
        
class veri_hinge_Fg_wo_U(veri_hinge_FG_wo_U):
    # segment verifier without control input constraints
    def __init__(self, model, Case, S_list):
        super().__init__(model, Case, S_list)
        self.W_out_list = [seg.W_out for seg in self.segs]
        self.r_out_list = [seg.r_out for seg in self.segs]
        self.SMT_flag = False
    
    def min_Lf_hinge(self, reverse_flag=False):
        return super().min_Lf_hinge(reverse_flag)
    
    def Farkas_lemma(self, reverse_flag=False, SMT_flag=False):
        super().Farkas_lemma(SMT_flag)   
        
    def verification(self, reverse_flag):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            veri_flag, ce = self.min_Lf_hinge(reverse_flag)
            if veri_flag:
                return True, None
            else:
                veri_flag, ce = self.Farkas_lemma(reverse_flag, SMT_flag=True)
        return veri_flag, ce

class veri_seg_fG_wo_U(veri_seg_FG_wo_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.SMT_flag = False
        
    def zero_Lg(self):
        return super().zero_Lg()
    
    def min_Lf(self, reverse_flag=False, SMT_flag=False):
        if not SMT_flag:
            return super().min_Lf(reverse_flag)
        else:
            # TODO: add the SMT solver for the nonlinear case
            pass
    
    def verification(self, reverse_flag=False):
        check_Lg = self.zero_Lg()
        if check_Lg:
            return True, None
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            veri_flag, ce = self.min_Lf(reverse_flag, SMT_flag=False)
            return veri_flag, ce
        
class veri_seg_fG_with_interval_U(veri_seg_FG_with_interval_U):
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
    
    def min_Lf_interval(self, reverse_flag=False):
        return super().min_Lf_interval(reverse_flag)
    
    def min_Lf_interval_SMT(self, reverse_flag=False):
        pass
    
    def verification(self, reverse_flag=False):
        check_Lg = self.zero_Lg()
        if check_Lg:
            return True, None
        elif not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            return self.min_Lf_interval_SMT(reverse_flag)
            
class veri_hinge_fG_wo_U(veri_hinge_FG_wo_U):
    # segment verifier without control input constraints
    def __init__(self, model, Case, S_list):
        super().__init__(model, Case, S_list)
        self.W_out_list = [seg.W_out for seg in self.segs]
        self.r_out_list = [seg.r_out for seg in self.segs]
        self.SMT_flag = False
    
    def same_sign_Lg(self):
        return super().same_sign_Lg()
    
    def min_Lf_hinge(self, reverse_flag=False):
        return super().min_Lf_hinge(reverse_flag)
    
    def Farkas_lemma(self, reverse_flag=False, SMT_flag=False):
        # TODO: add the SMT solver for the nonlinear case
        pass
    
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            return self.Farkas_lemma(reverse_flag, SMT_flag=True)

class veri_seg_Nfg_wo_U(veri_seg_Fg_wo_U):
    # segment verifier without control input constraints
    # Nonlinear f(x) and g(x)u
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.SMT_flag = False
        
    def zero_NLg(self):
        return super().zero_NLg()
    
    def min_NLfg(self, reverse_flag=False, SMT_flag=False):
        if not SMT_flag:
            return super().min_LFNLg(reverse_flag)
        else:
            # TODO: add the SMT solver for the nonlinear case
            pass
    
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            check_Lg = self.zero_NLg()
            if check_Lg:
                return True, None
            veri_flag, ce = self.min_NLfg(reverse_flag, SMT_flag=self.SMT_flag)
            return veri_flag, ce

class veri_hinge_Nfg_wo_U(veri_hinge_Fg_wo_U):
    # segment verifier without control input constraints
    # Nonlinear f(x) and g(x)u
    def __init__(self, model, Case, S_list):
        super().__init__(model, Case, S_list)
    
    def Farkas_lemma(self, reverse_flag=False, SMT_flag=False):
        super().Farkas_lemma(reverse_flag, SMT_flag)  
        
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            veri_flag, ce = self.Farkas_lemma(reverse_flag, SMT_flag=self.SMT_flag)
            return veri_flag, ce

class veri_seg_Nfg_with_interval_U(veri_seg_FG_with_interval_U):
    # segment verifier without control input constraints
    # Nonlinear f(x) and g(x)u
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.D = np.max(np.abs(self.Case.CTRLDOM), axis=1)
        self.threshold = 0
        self.SMT_flag = False

    def min_NLf_interval(self, reverse_flag=False):
        prog = MathematicalProgram()
        
        x = prog.NewContinuousVariables(self.dim, "x")
        prog = self.XS(prog, x)

        # Add linear constraints
        prog.AddLinearConstraint(self.W_out @ x + self.r_out == 0)
        # Add cost function
        fx = self.Case.f_x(x)
        Lfb = (self.W_o[self.index_o] @ fx ).flatten()[0]
        threshold = self.W_out @ self.Case.g_x(x) @ self.D
        if reverse_flag:
            LC = prog.AddCost(-Lfb-threshold)
        else:
            LC = prog.AddCost(Lfb+threshold)
        
        # Now solve the program.
        result = Solve(prog)
        
        if result.get_optimal_cost() + self.thrshold < 0:
            return False, result.GetSolution(x)
        else:
            return True, None
    
    def Farkas_lemma(self, reverse_flag=False, SMT_flag=False):
        # TODO: add the SMT solver for the nonlinear case
        pass
        
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            veri_flag, veri_res_x = self.min_NLf_interval(reverse_flag)
            if veri_flag:
                return True, None
        veri_flag, veri_res_x = self.Farkas_lemma(reverse_flag, SMT_flag=True)
        return veri_flag, veri_res_x
        
class veri_seg_Nfg_with_con_U(veri_seg_FG_with_con_U):
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.SMT_flag = False
        
    def Farkas_lemma(self, reverse_flag=False, SMT_flag=False):
        # TODO: add the SMT solver for the nonlinear case
        pass
    
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            return self.Farkas_lemma(reverse_flag, SMT_flag=True)
        
class veri_hinge_Nfg_cons_U(veri_hinge_Nfg_wo_U):
    # segment verifier without control input constraints
    # Nonlinear f(x) and g(x)u
    def __init__(self, model, Case, S_list):
        super().__init__(model, Case, S_list)
        self.SMT_flag = False
    
    def Farkas_lemma(self, SMT_flag=False):
        # TODO: add the SMT solver for the nonlinear case
        pass 
    
    def verification(self, reverse_flag=False):
        if not self.SMT_flag:
            return super().verification(reverse_flag)
        else:
            return self.Farkas_lemma(reverse_flag, SMT_flag=True)