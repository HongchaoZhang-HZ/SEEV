from utils import *

class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        '''
        Initializes a NeuralNetwork object.

        Parameters:
        architecture (list): A list of tuples representing the architecture of the neural network.
                                Each tuple contains the activation function and the size of the layer.

        Raises:
        ValueError: If an unknown activation function is provided.
        
        # Example usage
        architecture = [('relu', 64), ('relu', 32), ('linear', 1)]
        model = NeuralNetwork(architecture)
        print(model)
        '''
        super(NeuralNetwork, self).__init__()
        self.layers = nn.ModuleList()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for i, (activation, layer_size) in enumerate(architecture):
            if i == 0:
                input_size = layer_size
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
    