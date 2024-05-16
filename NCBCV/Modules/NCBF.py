from sympy import li
from NCBCV.Modules.utils import *
from NCBCV.Modules.NNet import NeuralNetwork as NNet
from torch import optim
from NCBCV.Cases import Cases

class NCBF(NNet):
    def __init__(self, architecture:list, Case:Cases):
        super().__init__(architecture)
        self.case = Case

    

