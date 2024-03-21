from Modules import *

# Given a linear expression of a ReLU NN (Activation set $S$), 
# return a set of linear constraints to formulate $\mathcal{X}(S)$

# Region of Activation (RoA) is the set of points that are activated by a ReLU NN
def RoA(S):
    # Initialize the set of constraints
    X = []
    
    # Get the number of layers in the NN
    num_layers = len(S)
    
    # Iterate over the layers of the NN
    for i in range(num_layers):
        # Get the current layer
        layer = S[i]
        
        # Get the activation function and the number of neurons in the layer
        activation, num_neurons = layer
        
        # If the activation function is ReLU
        if activation == 'relu':
            # Iterate over the neurons in the layer
            for j in range(num_neurons):
                # Initialize the constraint for the neuron
                constraint = []
                
                # Iterate over the weights of the neuron
                for k in range(len(S[i-1][1])):
                    # Get the weight of the neuron
                    weight = S[i-1][1][k][j]
                    
                    # Add the weight to the constraint
                    constraint.append(weight)
                
                # Add the bias of the neuron to the constraint
                constraint.append(S[i-1][1][-1][j])
                
                # Add the constraint to the set of constraints
                X.append(constraint)
    
    # Return the set of constraints
    return X

# Given a activation set $S$, return the linear expression of the output of the ReLU NN
def LinearExp(S):
    # Input: S: Activation set of a ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    pass

def solver():
    # Input: X: Linear expression of the output of the ReLU NN
    # Output: X: Linear expression of the output of the ReLU NN
    pass
