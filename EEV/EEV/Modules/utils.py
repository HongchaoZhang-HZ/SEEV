import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dataclasses
from typing import Optional, Tuple, Union
import numpy as np


def generate_samples(domain, num_samples):
    # Get the dimensions of the domain
    num_dimensions = len(domain)

    # Convert once so scaling stays entirely in Torch while preserving the
    # dtype selected by the previous NumPy/Torch operation. Torch keeps
    # integer bounds in float32 arithmetic, while the legacy mixed operation
    # promoted them to float64, so make that promotion explicit.
    domain_array = np.array(domain)
    if np.issubdtype(domain_array.dtype, np.integer):
        domain_array = domain_array.astype(np.float64)
    domain = torch.as_tensor(domain_array)
    lows = domain[:, 0]
    spans = domain[:, 1] - domain[:, 0]

    # Generate random samples
    samples = torch.rand((num_samples, num_dimensions))

    # Scale the samples to the domain (Torch-only arithmetic)
    samples = samples * spans + lows

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
   
# Example: Generate the samples and visualize them
# samples = generate_samples([(-1,1),(-1,1)], 100)
# visualize_samples(samples)
