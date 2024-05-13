from sympy import li
from Phase1_Scalability.Modules.utils import *
from NNet import NeuralNetwork as NNet
from torch import optim
from Cases import Cases

class NCBF(NNet):
    def __init__(self, architecture:list, Case:Cases):
        super().__init__(architecture)
        self.case = Case

    

