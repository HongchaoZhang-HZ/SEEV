from collections import deque
from .utils import *

class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        '''
        Initializes a NeuralNetwork object.
        Parameters:
        architecture (list): A list of tuples representing the architecture of the neural network.
                                Each tuple contains the activation function and the size of the layer.
        '''
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i, (activation, layer_size) in enumerate(architecture):
            if i == 0:
                input_size = layer_size
                continue
            else:
                input_size = architecture[i-1][1]
            
            self.layers.append(nn.Linear(input_size, layer_size))
            
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'linear':
                self.layers.append(nn.Identity())
            else:
                raise ValueError(f"Unknown activation function: {activation}")
                
        self.to(self.device)
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def merge_last_n_layers(self, n):
        """
        Merges the last `n` linear layers.
        n (int): Number of layers to merge.
        """
        # Verify that `n` is within the bounds of layers available
        linear_indices = [i for i, layer in enumerate(self.layers) if isinstance(layer, nn.Linear)]

        if n < 2 or n > len(linear_indices):
            raise ValueError(f"Number of layers to merge should be between 2 and {len(linear_indices)}")

        # Get the indices of the linear layers to be merged
        indices_to_merge = linear_indices[-n:]
        
        # Initialize the merged weights and biases using the last layer
        last_layer_index = indices_to_merge[-1]
        merged_weight = self.layers[last_layer_index].weight.data
        merged_bias = self.layers[last_layer_index].bias.data

        # Loop backward through the layers to be merged
        for index in reversed(indices_to_merge[:-1]):
            prev_layer = self.layers[index]
            merged_weight = torch.matmul(merged_weight, prev_layer.weight.data).to('cpu')
            merged_bias = merged_bias.to('cpu') + torch.matmul(merged_weight.to('cpu'), prev_layer.bias.data.to('cpu')).to('cpu')

        # Create the new merged layer
        new_layer = nn.Linear(merged_weight.shape[1], merged_weight.shape[0])
        new_layer.weight.data = merged_weight
        new_layer.bias.data = merged_bias

        # Remove old layers and replace them with the new merged layer
        remaining_layers = list(self.layers)[:indices_to_merge[0]]
        remaining_layers.append(new_layer)

        # Update ModuleList
        self.layers = nn.ModuleList(remaining_layers)
