import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Status import NeuronStatus, NetworkStatus

class NSBasic(NetworkStatus):
    def __init__(self, network):
        super().__init__(network)
        
    