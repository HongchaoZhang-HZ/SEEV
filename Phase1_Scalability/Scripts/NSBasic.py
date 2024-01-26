import re
import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Scripts.Status import NeuronStatus, NetworkStatus

class NSBasic:
    '''Class to store the basic functions for the network status
    Variables:
    set_P: the set of neuron who preactivation input is positive
    set_N: the set of neuron who preactivation input is negative
    set_Z: the set of neuron who preactivation input is zero
    set_U: the set of neuron who preactivation input is unknown
    Features:
    neuron_layer_is_in_set: checks if the neuron is in the set
    neuron_idx_is_in_layer: checks if the neuron index is in the layer
    is_in_set: checks if the neuron is in the set
    add_to_set: adds the neuron to the set
    remove_from_set: removes the neuron from the set
    update_set: updates the set
    move: moves the neuron from one set to another
    display: prints the set
    '''
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
                value = 1
            elif set_type == 'N':
                stype = self.set_N
                value = -1
            elif set_type == 'Z':
                stype = self.set_Z
                value = 0
            elif set_type == 'U':
                stype = self.set_U
                value = -2
            if NeuronS.layer in stype.keys():
                NeuronS.set_status(value)
                stype[NeuronS.layer][NeuronS.neuron] = NeuronS
            else:
                NeuronS.set_status(value)
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
        
    def move(self, 
             from_set:str, 
             to_set:str, 
             NeuronS:NeuronStatus) -> None:
        self.remove_from_set(from_set, NeuronS)
        self.add_to_set(to_set, NeuronS)
        
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
    '''Class to store the network status for QBasic
    Variables:
    SOI: Set of Important Neurons
    Pcon: Constraints for the positive neurons
    Ncon: Constraints for the negative neurons
    Zcon: Constraints for the zero neurons
    Ucon: Constraints for the unknown neurons
    Features:
    get_network: gets the network to intialize the U set
    get_NS_input: gets the network status from the input value
    init_NS: initializes the network status
    init_SOI: initializes the set of important neurons
    update_SOI: updates the set of important neurons
    '''
    def __init__(self):
        super().__init__()
        self._SOI = {}
        self._Pcon = {}
        self._Ncon = {}
        self._Zcon = {}
        self._Ucon = {}
    
    def get_network(self, network:NNet) -> None:
        self.network = network
        for layer_idx, layer in enumerate(self.network.layers):
            list_of_neurons = {}
            if layer_idx % 2 == 0:
                num_neurons = layer.out_features
                for neuron_idx in range(num_neurons):
                    ns = NeuronStatus(int(layer_idx/2), int(neuron_idx), -2)
                    list_of_neurons[neuron_idx] = ns
                self.set_U[int(layer_idx/2)] = list_of_neurons
    
    def get_NS_input(self, input_value:torch.tensor, type:str) -> dict:
        '''Get the network status from the input value
        input_value: torch.tensor
        type: str (NI, NS, NSV)
        NI: Neuron Inputs, NS: Network Status, NSV: Network Status Values
        '''
        self._NStatus.get_netstatus_from_input(input_value)
        if type == 'NI':
            return self._NStatus.neuron_inputs
        elif type == 'NS':
            return self._NStatus.network_status
        elif type == 'NSV':
            return self._NStatus.network_status_values
    
    def init_NS(self, network) -> None:
        self._NStatus = NetworkStatus(network)
        self.get_network(network)
    
    @property
    def SOI(self):
        return self._SOI
    
    @SOI.setter
    def SOI(self, SOI:dict):
        self._SOI = SOI
        
    def init_SOI(self):
        pass
    
    def update_SOI(self):
        pass
    
    @property
    def Pcon(self):
        return self._Pcon
    
    @Pcon.setter
    def Pcon(self, Pcon):
        self._Pcon = Pcon
        
    @property
    def Ncon(self):
        return self._Ncon
    
    @Ncon.setter
    def Ncon(self, Ncon):
        self._Ncon = Ncon
        
    @property
    def Zcon(self):
        return self._Zcon
    
    @Zcon.setter
    def Zcon(self, Zcon):
        self._Zcon = Zcon
        
    @property
    def Ucon(self):
        return self._Ucon
    
    @Ucon.setter
    def Ucon(self, Ucon):
        self._Ucon = Ucon
    
if __name__ == '__main__':
    NS = NS()
    from Modules.NNet import NeuralNetwork as NNet
    network = NNet([('relu', 2), ('relu', 32), ('linear', 1)])
    NS.get_network(network)
    NS.display('U')
    
    
