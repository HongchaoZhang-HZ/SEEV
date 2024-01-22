import sys, os
sys.path.append(os.path.realpath(os.path.dirname(__file__)+"/.."))
from Modules.utils import *
from Modules.NNet import NeuralNetwork as NNet

class NeuronStatus:
    def __init__(self, layer:int, neuron:int, status:int):
        self.layer = layer
        self.neuron = neuron
        # status: -2: unknown, -1: negative, 0: zero, 1: positive
        self.status = status if status is not None else -2
        self.tol = 1e-4

    def get_id(self):
        return [self.layer, self.neuron]
    
    def get_status(self):
        return self.status

    def set_status(self, new_status) -> None:
        self.status = new_status
        
    def set_status_from_value(self, value) -> None:
        if value > self.tol:
            self.status = 1
        elif value < -self.tol:
            self.status = -1
        else:
            self.status = 0
        
    def display(self):
        print("Layer: ", self.layer, " Neuron: ", self.neuron, " Status: ", self.status)
        
        
class NetworkStatus:
    def __init__(self, network):
        self.network = network
        self.neuron_inputs = {}
        self.network_status = {}
        self.network_status_values = {}

    def set_layer_status(self, layer_idx, layer_status):
        for neuron_idx, neuron_status in enumerate(layer_status):
            if self.network_status == {}:
                self.network_status = [NeuronStatus(layer_idx, status_item, -2) for status_item in layer_status]
            else:
                for status_idx, status_item in enumerate(layer_status):
                    self.network_status[status_idx].set_status(status_item)
        
    def set_layer_status_from_value(self, layer_idx, input_value) -> list:
        layer_status = list([])
        for neuron_idx, neuron_input in enumerate(input_value):
            neuron_status = NeuronStatus(layer_idx, neuron_idx, -2)
            neuron_status.set_status_from_value(neuron_input)
            neuron_status.display()
            layer_status.append(neuron_status)
        return layer_status
    
    def get_neuron_inputs(self) -> dict:
        return self.neuron_inputs
    
    def display_layer_status(self, layer_status):
        print("Layer Status: ", [nstatus.status for nstatus in layer_status])

    def forward_propagation(self, input_value) -> None:
        x = input_value
        for layer_idx, layer in enumerate(self.network.layers):
            x = layer(x)
            # if layer is even, then it is a linear layer not activation layer
            if layer_idx % 2 == 0: # starting from 0
                self.neuron_inputs[int(layer_idx/2)] = x.tolist()
                layer_status = self.set_layer_status_from_value(layer_idx, x)
                self.network_status[int(layer_idx/2)] = layer_status  # type: ignore # Assign the first NeuronStatus object
                self.network_status_values[int(layer_idx/2)] = [nstatus.status for nstatus in layer_status]
        print('Propagation completed.')
        self.display_network_status_value()
        
    def display_network_status_value(self):
        print("Network Status: ", self.network_status_values)

if __name__ == '__main__':
    architecture = [('relu', 2), ('relu', 32), ('linear', 1)]
    model = NNet(architecture)
    NStatus = NetworkStatus(model)

    # Generate random input using torch.rand for the model
    input_size = model.layers[0].in_features
    random_input = torch.rand(input_size)
    x = random_input
    NStatus.forward_propagation(x)
    