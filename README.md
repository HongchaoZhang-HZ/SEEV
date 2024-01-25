# NCBCV


# Summary of Phase 1 Development
`pytest --md-report --md-report-flavor gfm --md-report-output md_report.md` to generate the markdown report [md_report](md_report.md). 

|                 filepath                 | $$\textcolor{#23d18b}{\tt{passed}}$$ | SUBTOTAL |
| ---------------------------------------- | --------------------------------: | -------: |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_BaBBasic.py}}$$ |   $$\textcolor{#23d18b}{\tt{4}}$$ | $$\textcolor{#23d18b}{\tt{4}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_NNet.py}}$$ |   $$\textcolor{#23d18b}{\tt{2}}$$ | $$\textcolor{#23d18b}{\tt{2}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_NSBasic.py}}$$ |  $$\textcolor{#23d18b}{\tt{15}}$$ | $$\textcolor{#23d18b}{\tt{15}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_Status.py}}$$ |   $$\textcolor{#23d18b}{\tt{6}}$$ | $$\textcolor{#23d18b}{\tt{6}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_utils.py}}$$ |   $$\textcolor{#23d18b}{\tt{2}}$$ | $$\textcolor{#23d18b}{\tt{2}}$$ |
| $$\textcolor{#23d18b}{\tt{TOTAL}}$$      |  $$\textcolor{#23d18b}{\tt{29}}$$ | $$\textcolor{#23d18b}{\tt{29}}$$ |

<!-- # Strucutre of the Project
```plaintext
NCBF
├── Cases.py
│   ── Different Cases
├── NNet.py 
``` -->

## Development Log: Status (Neural Network Status Tracker)

This script is designed for monitoring and analyzing the internal states of a neural network. It focuses on tracking the status of individual neurons across different layers during data processing.

### Key Features

#### NeuronStatus Class
- **Function**: Represents and manages the state of a single neuron.
- **Key Methods**:
  - Status determination based on neuron values.
  - Status display for individual neurons.

#### NetworkStatus Class
- **Function**: Handles the status of the entire neural network.
- **Key Methods**:
  - Updates and displays the statuses of neurons in each layer.
  - Processes input data through the network for real-time status monitoring.

### Usage
- The script sets up a neural network, processes a random input, and visualizes neuron states.
- Ideal for debugging and understanding neuron behaviors within network layers.

### Dependencies
- Relies on custom modules (`Modules.utils` and `Modules.NNet`) and PyTorch.


## Development Log: Neuron Status Management in Neural Networks
This script implements a system for managing and tracking the status of neurons in neural networks, aiding in network analysis and debugging.

### Key Features

#### NSBasic Class
- **Functionality**: Handles management of neuron status sets (Positive, Negative, Zero, Unknown).
- **Features**: 
  - Includes getters and setters for status set management.
  - Provides methods for adding, removing, and updating neuron statuses.

#### NS Class (Extension of NSBasic)
- **Enhancement**: Adds additional functionalities, currently unspecified.
- **Placeholders**: Includes `init_SOI()` and `update_SOI()` methods for future development.

### Usage
- The classes are designed to be integrated with neural network models for monitoring neuron statuses.
- They offer functionalities to dynamically manage and track the status of individual neurons across network layers.
- The script can be particularly useful for neural network debugging and detailed analysis of neuron behavior during model training and inference.

### Dependencies
- External modules: `Modules.utils`, `Modules.NNet`.
- Custom classes: `NeuronStatus`, `NetworkStatus` from `Scripts.Status`.
- The script is dependent on these modules for core functionalities like neural network architecture and utility functions.
