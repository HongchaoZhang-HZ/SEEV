import re
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Scripts.Status import NeuronStatus, NetworkStatus

class NSBasic:
    def __init__(self):
        self._set_P = {}
        self._set_N = {}
        self._set_Z = {}
        self._set_U = {}
        
    @property
    def set_P(self) -> dict:
        return self._set_P
        
    @property
    def set_N(self) -> dict:
        return self._set_N
    
    @property
    def set_Z(self) -> dict:
        return self._set_Z
    
    @property
    def set_U(self) -> dict:
        return self._set_U
    
    @set_P.setter
    def set_P(self, set_P) -> None:
        self._set_P = set_P
    
    @set_N.setter
    def set_N(self, set_N) -> None:
        self._set_N = set_N
        
    @set_Z.setter
    def set_Z(self, set_Z) -> None:
        self._set_Z = set_Z
        
    @set_U.setter
    def set_U(self, set_U) -> None:
        self._set_U = set_U
        
    def get_network(self, network) -> None:
        self.network = network
        for layer_idx, layer in enumerate(self.network.layers):
            list_of_neurons = {}
            if layer_idx % 2 == 0:
                num_neurons = layer.out_features
                for neuron_idx in range(num_neurons):
                    ns = NeuronStatus(int(layer_idx/2), int(neuron_idx), -2)
                    list_of_neurons[neuron_idx] = ns
                self.set_U[int(layer_idx/2)] = list_of_neurons
        
    def neuron_layer_is_in_set(self, 
                               set:dict, 
                               NeuronS:NeuronStatus) -> bool:
        if NeuronS.layer in set.keys():
            return True
        else:
            return False
        
    def neuron_idx_is_in_layer(self, 
                               layer:dict, 
                               NeuronS:NeuronStatus) -> bool:
        if NeuronS.neuron in layer.keys():
            return True
        else:
            return False
    
    def is_in_set(self, 
                  set_type:str, 
                  NeuronS:NeuronStatus) -> bool:
        if set_type == 'P':
            stype = self.set_P
        elif set_type == 'N':
            stype = self.set_N
        elif set_type == 'Z':
            stype = self.set_Z
        elif set_type == 'U':
            stype = self.set_U
        NeuronID = NeuronS.get_id()
        if NeuronID[0] in stype.keys() and NeuronID[1] in stype[NeuronID[0]].keys():
            return True
        else:
            return False
    
    def add_to_set(self, 
                   set_type:str, 
                   NeuronS: NeuronStatus) -> None:
        if not self.is_in_set(set_type, NeuronS):
            if set_type == 'P':
                stype = self.set_P
            elif set_type == 'N':
                stype = self.set_N
            elif set_type == 'Z':
                stype = self.set_Z
            elif set_type == 'U':
                stype = self.set_U
            if NeuronS.layer in stype.keys():
                stype[NeuronS.layer][NeuronS.neuron] = NeuronS
            else:
                stype[NeuronS.layer] = {NeuronS.neuron: NeuronS}
    
    def remove_from_set(self,
                        set_type:str,
                        NeuronS:NeuronStatus) -> None:
        if self.is_in_set(set_type, NeuronS):
            if set_type == 'P':
                stype = self.set_P
            elif set_type == 'N':
                stype = self.set_N
            elif set_type == 'Z':
                stype = self.set_Z
            elif set_type == 'U':
                stype = self.set_U
            del stype[NeuronS.layer][NeuronS.neuron]
    
    def update_set(self,
                   set_type:str,
                   NeuronS:NeuronStatus) -> None:
        if self.is_in_set(set_type, NeuronS):
            self.remove_from_set(set_type, NeuronS)
        self.add_to_set(set_type, NeuronS)
        
    def display(self, set_type:str) -> None:
        if set_type == 'P':
            stype = self.set_P
        elif set_type == 'N':
            stype = self.set_N
        elif set_type == 'Z':
            stype = self.set_Z
        elif set_type == 'U':
            stype = self.set_U
        for layer_idx in stype.keys():
            print("Layer: ", layer_idx)
            for neuron_idx in stype[layer_idx].keys():
                stype[layer_idx][neuron_idx].display()
        
class NS(NSBasic):
    def __init__(self):
        super().__init__()
        self._SOI = {}
    
    @property
    def SOI(self):
        return self._SOI
    
    @SOI.setter
    def SOI(self, SOI):
        self._SOI = SOI
        
    def init_SOI(self):
        pass
    
    def update_SOI(self):
        pass
    
if __name__ == '__main__':
    NS = NS()
    from Modules.NNet import NeuralNetwork as NNet
    network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
    NS.get_network(network)
    NS.display('U')
    
    
