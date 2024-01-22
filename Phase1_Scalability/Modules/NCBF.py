from Phase1_Scalability.Modules.utils import *
from NNet import NeuralNetwork as NNet
from torch import optim

class NCBF(NNet):
    def __init__(self, architecture=None, domain=None):
        self.architecture = architecture if architecture is not None else [('relu', 2), ('relu', 32), ('linear', 1)]
        super().__init__(self.architecture)
        self.DOMAIN = domain if domain is not None else [(-2, 2), (-2, 2)]
        self.DIM = len(self.DOMAIN)

    

