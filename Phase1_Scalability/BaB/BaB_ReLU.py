import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Modules.NCBF import NCBF
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
from Scripts.BaBBasic import QBasic, BaBBasic

class BaB_ReLU:
    def __init__(self, network):
        self._NStatus = NSBasic()
        self._NStatus.init_NS(network)
        self._QBasic = QBasic(network)
        self._BaBBasic = BaBBasic(self._NStatus, self._QBasic)
        self._best_solution = None
        self._best_cost = float('inf')
        self._initial_network_status = self._NStatus
        self._set_U = self._NStatus.set_U
        