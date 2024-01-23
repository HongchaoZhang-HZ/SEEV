# NCBCV


# Summary of Phase 1 Development

[md_report](md_report.md)
|                filepath                | $$\textcolor{#23d18b}{\tt{passed}}$$ | SUBTOTAL |
| -------------------------------------- | --------------------------------: | -------: |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_NNet.py}}$$ |   $$\textcolor{#23d18b}{\tt{2}}$$ | $$\textcolor{#23d18b}{\tt{2}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_Status.py}}$$ |   $$\textcolor{#23d18b}{\tt{6}}$$ | $$\textcolor{#23d18b}{\tt{6}}$$ |
| $$\textcolor{#23d18b}{\tt{Phase1\\_Scalability/test/test\\_utils.py}}$$ |   $$\textcolor{#23d18b}{\tt{2}}$$ | $$\textcolor{#23d18b}{\tt{2}}$$ |
| $$\textcolor{#23d18b}{\tt{TOTAL}}$$    |  $$\textcolor{#23d18b}{\tt{10}}$$ | $$\textcolor{#23d18b}{\tt{10}}$$ |




## Status (Neural Network Status Tracker)

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
