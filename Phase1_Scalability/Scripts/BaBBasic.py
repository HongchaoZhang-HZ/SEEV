# import sys, os
# sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS

class QBasic(NS):
    def __init__(self, network):
        super().__init__()
        self.init_NS(network)
        

class BaBBasic():
    def __init__(self, network):
        self.network = network
        self.NStatus = NetworkStatus(network)
        self.NS = NS()
    
    @property
    def NStatus(self):
        return self.NStatus
    
    @property
    def NS(self):
        return self.NS
    
    @NStatus.setter
    def NStatus(self, NStatus):
        self.NStatus = NStatus
        
    @NS.setter
    def NS(self, NS):
        self.NS = NS
