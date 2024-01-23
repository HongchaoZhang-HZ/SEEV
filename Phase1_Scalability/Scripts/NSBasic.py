import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *

class NSBasic:
    def __init__(self):
        self.set_P = {}
        self.set_N = {}
        self.set_Z = {}
        self.set_U = {}
        
    @property
    def set_P(self):
        return self.set_P
        
    @property
    def set_N(self):
        return self.set_N
    
    @property
    def set_Z(self):
        return self.set_Z
    
    @property
    def set_U(self):
        return self.set_U
    
    @set_P.setter
    def set_P(self, set_P):
        self.set_P = set_P
    
    @set_N.setter
    def set_N(self, set_N):
        self.set_N = set_N
        
    @set_Z.setter
    def set_Z(self, set_Z):
        self.set_Z = set_Z
        
    @set_U.setter
    def set_U(self, set_U):
        self.set_U = set_U
        
class NS(NSBasic):
    def __init__(self):
        super().__init__()
        self.SOI = {}
    
    @property
    def SOI(self):
        return self.SOI
    
    @SOI.setter
    def SOI(self, SOI):
        self.SOI = SOI
        
    def init_SOI(self):
        pass
    
    def update_SOI(self):
        pass
    
    