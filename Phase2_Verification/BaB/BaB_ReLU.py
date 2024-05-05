from calendar import c
import sys, os
from tkinter import W

from sympy import false
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Modules.NCBF import NCBF
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
from Scripts.BaBBasic import QBasic, BaBBasic

class ReLU_Q(QBasic):
    def __init__(self, network:NNet):
        super().__init__(network)
        self.tol = 1e-5
        # Use random init for now, replace with SOI later
        self.random_init()
        
    def random_init(self):
        input_size = self._NStatus._net_size[0]
        _ = self.Activation(input_size)
        
    def get_weight_bias_from_mask(self, 
                   layer_idx: int, 
                   neuron_idx: int, 
                   mask: dict) -> (torch.tensor, torch.tensor):
        
        W_bar = torch.tensor([1])
        r_bar = torch.tensor([0])
        return W_bar, r_bar
        
    def get_weight_bias(self, 
                   layer_idx: int, 
                   neuron_idx: int) -> (torch.tensor, torch.tensor):
        W_bar, r_bar = self.get_weight_bias_from_mask(layer_idx, 
                                                      neuron_idx, 
                                                      self.mask)
        return W_bar, r_bar
        
    def compute_pwl_output(self) -> (torch.tensor, torch.tensor):
        W_bar, r_bar = self.get_weight_bias(len(self.net_size), 0)
        return W_bar, r_bar
        
    def update_con_set(self) -> (dict, dict, dict):
        P_con = {}
        N_con = {}
        Z_con = {}
        for layer_idx in range(len(self.net_size)):
            for neuron_idx in self.net_size:
                neuron_status = self._mask[layer_idx][neuron_idx]
                W_bar, r_bar = self.get_weight_bias(layer_idx, neuron_idx)
                if neuron_status == 1:
                    P_con[(layer_idx, neuron_idx)] = (W_bar, r_bar)
                elif neuron_status == -1:
                    N_con[(layer_idx, neuron_idx)] = (W_bar, r_bar)
                elif neuron_status == 0:
                    Z_con[(layer_idx, neuron_idx)] = (W_bar, r_bar)
                else:
                    raise ValueError('Neuron status is not valid')
        return P_con, N_con, Z_con
    
    def formulate_constraints(self, update_flag:bool=false) -> dict:
        # Check if the con_sets are updated
        if update_flag:
            self.P_con, self.N_con, self.Z_con = self.update_con_set()
        if self.P_con == {} and self.N_con == {} and self.Z_con == {}:
            self.P_con, self.N_con, self.Z_con = self.update_con_set()
        # Formulate the constraints
        constraints = {}
        for layer_idx in self.P_con.keys():
            for neuron_idx in self.P_con[layer_idx].keys():
                w, b = self.P_con[layer_idx][neuron_idx]
                # TODO: Formulate the constraints
                # w*x + b > self.tol
                constraints[layer_idx][neuron_idx] = constraint(w, b, self.tol, '>')   
        for layer_idx in self.N_con.keys():
            for neuron_idx in self.N_con[layer_idx].keys():
                w, b = self.N_con[layer_idx][neuron_idx]
                # TODO: Formulate the constraints
                # w*x + b < -self.tol
                constraints[layer_idx][neuron_idx] = constraint(w, b, self.tol, '<')                
        for layer_idx in self.Z_con.keys():
            for neuron_idx in self.Z_con[layer_idx].keys():
                w, b = self.Z_con[layer_idx][neuron_idx]
                # TODO: Formulate the constraints
                # w*x + b almost equal to 0
                constraints[layer_idx][neuron_idx] = constraint(w, b, self.tol, '=')
        return constraints
    
    def prepare_Q(self):
        # Now the Q is initialized, next we update X and V
        self.P_con, self.N_con, self.Z_con = self.update_con_set()
        # self.update_X()
        constraints = self.formulate_constraints()
        self.X(constraints)
        # self.update_V()
        self.W_overline, self.r_overline = self.compute_pwl_output()
        self.V(self.W_overline)
    
    
class BaB_ReLU:
    def __init__(self, network: NNet):
        self._NStatus = NSBasic()
        self._NStatus.init_NS(network)
        self._QBasic = QBasic(network)
        self._BaBBasic = BaBBasic(self._NStatus, self._QBasic)
        self._best_solution = None
        self._best_cost = float('inf')
        self._initial_network_status = self._NStatus
        self._set_U = self._NStatus.set_U
        