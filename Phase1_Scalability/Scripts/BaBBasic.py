import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS

class QBasic(NS):
    '''Class to store the basic functions for Q tuple
    Variables:
    _NStatus: NetworkStatus, network status
    _net_size: list, size of the network
    _mask: dict, mask for the network
    Features:
    net_size: returns the size of the network
    update_Q_from_NS: updates the Q from NS
    update_mask: updates the mask from the Q
    '''
    def __init__(self, network):
        super().__init__()
        self.init_NS(network)
        self._net_size = self.net_size()
        self._mask = self.init_mask()
        
    def net_size(self):
        size = [self.network.layers[2*layer_idx].out_features
                for layer_idx in range(int(len(self.network.layers)/2))]
        return size
        
    def update_Q_from_NS(self):
        check_dict = self._NStatus.network_status_values
        if check_dict == {}:
            raise ValueError("Network status has not been initialized")
        for layer_idx in check_dict.keys():
            for neuron_idx in range(len(check_dict[layer_idx])):
                if check_dict[layer_idx][neuron_idx] == 1:
                    set_type = 'P'
                elif check_dict[layer_idx][neuron_idx] == -1:
                    set_type = 'N'
                elif check_dict[layer_idx][neuron_idx] == 0:
                    set_type = 'Z'
                else:
                    set_type = 'U'
                self.remove_from_set('U', self._NStatus.network_status[layer_idx][neuron_idx])
                self.add_to_set(set_type, self._NStatus.network_status[layer_idx][neuron_idx])
        
    @property
    def mask(self):
        return self._mask
    
    @mask.setter
    def mask(self, new_mask):
        for layer_idx in range(len(self._net_size)):
            if new_mask[layer_idx].shape[0] != self._net_size[layer_idx]:
                raise ValueError("Mask shape does not match the input shape of the layer")
        self._mask = new_mask
    
    def init_mask(self) -> dict:
        mask = {}
        for layer_idx in range(len(self._net_size)):
            mask[layer_idx] = -2*torch.ones(self._net_size[layer_idx], 
                                            dtype=torch.int8).to(self.network.device)
        return mask
    
    def update_mask(self) -> None:
        for layer_idx in self.set_P.keys():
            for neuron_idx in self.set_P[layer_idx].keys():
                self._mask[layer_idx][neuron_idx] = 1
        for layer_idx in self.set_N.keys():
            for neuron_idx in self.set_N[layer_idx].keys():
                self._mask[layer_idx][neuron_idx] = -1
        for layer_idx in self.set_Z.keys():
            for neuron_idx in self.set_Z[layer_idx].keys():
                self._mask[layer_idx][neuron_idx] = 0
        for layer_idx in self.set_U.keys():
            for neuron_idx in self.set_U[layer_idx].keys():
                self._mask[layer_idx][neuron_idx] = -2
                
    def Activation(self, x:torch.tensor) -> dict:
        '''Function to get the value of the function
        x: torch.tensor
        '''
        self._NStatus.get_netstatus_from_input(x)
        self.update_Q_from_NS()
        self.update_mask()
        return self._mask
    
    def V(self, x:torch.tensor) -> torch.tensor:
        '''Function to get the derivative of NNet given Q
        x: torch.tensor
        '''
        pass
        
class BaBBasic():
    def __init__(self, network):
        self.network = network
        self.NStatus = NetworkStatus(network)
        # Q tuple is node
        self._Q = QBasic(network)
        # ToDo: Bound of Q
        self._X = {}
        # ToDo: Derivative of Q
        self._V = {}
        
    @property
    def Q(self):
        return self._Q
    
    @property
    def X(self):
        return self._X
    
    @X.setter
    def X(self, Q):
        pass
    
    @property
    def V(self):
        return self._V
    
    @V.setter
    def V(self, Q):
        pass
    
    
    

if __name__ == '__main__':
    Q = QBasic(NNet([('relu', 2), ('relu', 32), ('linear', 1)]))
    input_size = Q._NStatus.network.layers[0].in_features
    input = torch.rand(input_size)
    Q._NStatus.get_netstatus_from_input(input)
    Q.update_Q_from_NS()