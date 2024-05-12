import sys, os
import time
from pytest import fail
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Verifier.VeriUtil import *
from Verifier.LinearVeri import *
from Verifier.NLVeri import *

from Modules.NNet import NeuralNetwork as NNet
from Cases.Darboux import Darboux
from Cases.ObsAvoid import ObsAvoid
from Cases.LinearSatellite import LinearSat
from Scripts.Search import Search

import warnings
warnings.filterwarnings("ignore")


class Verifier(verifier_basic):
    def __init__(self, model, Case, 
                 unstable_neurons_set, 
                 pair_wise_hinge, 
                 ho_hinge, VNBC_flag=False):
        super().__init__(model, Case)
        self.unstable_neurons_set = unstable_neurons_set
        self.pair_wise_hinge = pair_wise_hinge
        self.ho_hinge = ho_hinge
        self.diagnose()
        self.feasibility_only = False
        
        # T - Barrier Certificate Verification, 
        # F - Control Barrier Function Verification
        self.VNBC_flag = VNBC_flag 
        
    def diagnose(self):
        if self.is_gx_linear:
            type_g = 'G'
        else:
            type_g = 'g'
        if self.is_fx_linear:
            type_f = 'F'
        else:
            type_f = 'f'
        if self.is_u_cons:
            type_u = 'U'
            if self.is_u_cons_interval:
                type_u += 'i'
            else:
                type_u += 'c'
        else:
            type_u = 'N'
        self.type = type_f + type_g + type_u
        
    def seg_verifier(self, S):
        if self.type == 'FGN':
            return veri_seg_FG_wo_U(self.model, self.Case, S)
        elif self.type == 'fGN':
            return veri_seg_fG_wo_U(self.model, self.Case, S)
        elif self.type == 'FgN':
            return veri_seg_Fg_wo_U(self.model, self.Case, S)
        elif self.type == 'fgN':
            return veri_seg_Nfg_wo_U(self.model, self.Case, S)
        elif self.type == 'FGUi':
            return veri_seg_FG_with_interval_U(self.model, self.Case, S)
        elif self.type == 'fGUi':
            return veri_seg_fG_with_interval_U(self.model, self.Case, S)
        elif self.type == 'FgUi' or self.type == 'fgUi':
            return veri_seg_Nfg_with_interval_U(self.model, self.Case, S)
        elif self.type == 'FGUc':
            return veri_seg_FG_with_con_U(self.model, self.Case, S)
        elif self.type == 'FgUc' or self.type == 'fGUc' or self.type == 'fgUc':
            return veri_seg_Nfg_with_con_U(self.model, self.Case, S)
        
    def hinge_verifier(self, S):
        if self.type == 'FGN':
            return veri_hinge_FG_wo_U(self.model, self.Case, S)
        elif self.type == 'fGN':
            return veri_hinge_fG_wo_U(self.model, self.Case, S)
        elif self.type == 'FgN':
            return veri_hinge_Fg_wo_U(self.model, self.Case, S)
        elif self.type == 'fgN':
            return veri_hinge_Nfg_wo_U(self.model, self.Case, S)
        else:
            return veri_hinge_Nfg_cons_U(self.model, self.Case, S)
        
    def seg_verification(self, unstable_neurons_set, reverse_flag=False, SMT_flag=False):
        for S in unstable_neurons_set:
            seg_verifier = self.seg_verifier(S)
            if SMT_flag:
                seg_verifier.SMT_flag = True
            veri_flag, ce = seg_verifier.verification(reverse_flag, self.feasibility_only)
            if not veri_flag:
                print('Verification failed!')
                print('Segment counter example', ce)
                return False, ce
        return True, None
    
    def hinge_verification(self, hinge_set, reverse_flag=False, SMT_flag=False):
        for hinge in hinge_set:
            hinge_verifier = self.hinge_verifier(hinge)
            if SMT_flag:
                hinge_verifier.SMT_flag = True
            veri_flag, ce = hinge_verifier.verification(reverse_flag)
            if not veri_flag:
                print('Verification failed!')
                print('Hinge counter example', ce)
                return False, ce
        return True, None
    
    def Verification(self, reverse_flag=False, SMT_flag=False):
        veri_flag_seg, ce_seg = self.seg_verification(self.unstable_neurons_set, reverse_flag, SMT_flag)
        if not veri_flag_seg:
            return False, ce_seg
        veri_flag_hinge, ce_hinge = self.hinge_verification(self.pair_wise_hinge, reverse_flag, SMT_flag)
        if not veri_flag_hinge:
            return False, ce_hinge
        veri_flag_ho, ce_ho = self.hinge_verification(self.ho_hinge, reverse_flag, SMT_flag)
        if not veri_flag_ho:
            return False, ce_ho
        print('Verification passed!')
        return True, None

    
    