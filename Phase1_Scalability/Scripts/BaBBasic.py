import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Status import NeuronStatus, NetworkStatus
from NSBasic import NSBasic, NS

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
