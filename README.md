# NCBCV


# Summary of Phase 1 Development

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
