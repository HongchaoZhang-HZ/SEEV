import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet
from Scripts.Status import NeuronStatus, NetworkStatus
from Scripts.NSBasic import NSBasic, NS

class QBasic(NS):
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
            mask[layer_idx] = -2*torch.ones(self._net_size[layer_idx]).to(self.network.device)
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

if __name__ == '__main__':
    Q = QBasic(NNet([('relu', 2), ('relu', 32), ('linear', 1)]))
    input_size = Q._NStatus.network.layers[0].in_features
    input = torch.rand(input_size)
    Q._NStatus.get_netstatus_from_input(input)
    Q.update_Q_from_NS()