import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS
import PARA as p
from Foundation.Function import RoA, LinearExp

class Enumerate:
    def __init__(self) -> None:
        self.zero_tol = p.zero_tol
        pass
    
    