import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_samples(domain, num_samples):
    # Get the dimensions of the domain
    num_dimensions = len(domain)

    # Convert domain to numpy array
    domain = np.array(domain)

    # Generate random samples
    samples = torch.rand((num_samples, num_dimensions))

    # Scale the samples to the domain
    samples = samples * (domain[:, 1] - domain[:, 0]) + domain[:, 0]

    return samples

def visualize_samples(samples):
    # Extract the x and y coordinates from the samples
    x = samples[:, 0]
    y = samples[:, 1]

    # Plot the samples
    plt.scatter(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Generated Samples')
    plt.show()
   

# Generate the samples
# samples = generate_samples([(-1,1),(-1,1)], 100)
# visualize_samples(samples)
