from collections import deque

from torch import device
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
    
    def load_state_dict_from_sequential(self, sequential:nn.Sequential):
        ''' Adjusts and loads state dictionary from a sequential model '''
        sequential_state_dict = sequential.state_dict()
        custom_state_dict = self.state_dict()

        # Map from sequential keys to custom model keys
        # Here we assume every other layer in custom model is a Linear layer
        translation_map = {f'{int(i/2)*2}.{suffix}': f'layers.{i}.{suffix}' 
                           for i in range(0, len(sequential_state_dict), 2) 
                           for suffix in ['weight', 'bias']}

        # Create new state dictionary with translated keys
        new_state_dict = {}
        for seq_key, value in sequential_state_dict.items():
            if seq_key in translation_map:
                new_state_dict[translation_map[seq_key]] = value
            else:
                raise KeyError(f"Key {seq_key} not found in the translation map. Check layer alignment.")

        # Load the translated state dictionary
        self.load_state_dict(new_state_dict, strict=True)
    
    def wrapper_load_state_dict(self, trained_state_dict):
        # Get the keys from the current model's state dict
        model_state_dict = self.state_dict()
        model_keys = list(model_state_dict.keys())

        # Get the keys from the trained model's state dict
        trained_keys = list(trained_state_dict.keys())

        # Ensure both lists have the same length (or adjust accordingly)
        if len(model_keys) != len(trained_keys):
            raise ValueError("Mismatch in the number of layers or keys between models.")

        # Create a mapping between old (trained) and new (current) model keys
        key_mapping = {old_key: new_key for old_key, new_key in zip(trained_keys, model_keys)}

        # Create a new state dict with renamed keys
        renamed_state_dict = {key_mapping[old_key]: value for old_key, value in trained_state_dict.items() if old_key in key_mapping}

        return renamed_state_dict
    
    def merge_last_n_layers(self, n):
        """
        Merges the last `n` linear layers.
        n (int): Number of layers to merge.
        """
        # Verify that `n` is within the bounds of layers available
        print(self.to('cpu').forward(torch.tensor([1,0,0,0,0.0,0.0]).to('cpu')))
        
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
        device = 'cpu'
        merged_weight = merged_weight.to(device)
        merged_bias = merged_bias.to(device)

        # Loop through the indices of layers to merge in reverse order
        for index in reversed(indices_to_merge[:-1]):
            prev_layer = self.layers[index]
            prev_weight = prev_layer.weight.data.to(device)
            prev_bias = prev_layer.bias.data.to(device)

            # Temporary storage for the current weight calculation
            temp_weight = torch.matmul(merged_weight, prev_weight)

            # Update the bias calculation
            temp_bias = merged_bias + torch.matmul(merged_weight, prev_bias)

            # Update merged_weight and merged_bias
            merged_weight = temp_weight
            merged_bias = temp_bias

        # Create the new merged layer
        new_layer = nn.Linear(merged_weight.shape[1], merged_weight.shape[0])
        new_layer.weight.data = merged_weight
        new_layer.bias.data = merged_bias

        # Remove old layers and replace them with the new merged layer
        remaining_layers = list(self.layers)[:indices_to_merge[0]]
        remaining_layers.append(new_layer)
        remaining_layers.append(torch.nn.Identity())
        # Update ModuleList
        self.layers = nn.ModuleList(remaining_layers)
        print(self.to('cpu').forward(torch.tensor([1,0,0,0,0.0,0.0]).to('cpu')))
