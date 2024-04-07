import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))

from Scripts.SearchInit import *
from Scripts.SearchInit import SearchInit

class Search(SearchInit):
    def __init__(self, model, case=None) -> None:
        super().__init__(model, case)
        
    def specify_point(self, safe_point:torch.tensor, unsafe_point:torch.tensor):
        S_init = Search.initialization(safe_point, unsafe_point)
        self.S_init = S_init
        
        
    