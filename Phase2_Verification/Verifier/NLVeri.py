from LinearVeri import *

class veri_seg_Nfg_wo_U(veri_seg_FG_wo_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        
    def zero_NLg(self, x):
        pass
    
    def min_NLf(self, reverse_flat=False):
        pass

class veri_seg_Nfg_with_interval_U(veri_seg_FG_with_interval_U):
    # segment verifier without control input constraints
    # Linear Fx and Gu
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        self.D = np.max(np.abs(self.Case.CTRLDIM))
        self.threshold_interval()

    def min_NLf_interval(self, reverse_flat=False):
        pass
        
class veri_seg_Nfg_with_con_U(veri_seg_FG_with_con_U):
    def __init__(self, model, Case, S):
        super().__init__(model, Case, S)
        
        