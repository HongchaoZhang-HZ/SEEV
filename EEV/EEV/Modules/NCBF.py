from sympy import li
from EEV.Modules.utils import *
from EEV.Modules.NNet import NeuralNetwork as NNet
from torch import optim
from EEV.Cases import Cases

class NCBF(NNet):
    def __init__(self, architecture:list, Case:Cases):
        super().__init__(architecture)
        self.case = Case

    

