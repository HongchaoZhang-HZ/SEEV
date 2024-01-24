# Neural Network Status Management System

## Overview
This script implements a system for managing and displaying the status of neurons within a neural network. It consists of two main classes, `NSBasic` and `NS`, and a main execution block that demonstrates their usage with a neural network.

## Classes

### NSBasic
- **Purpose**: To handle sets of neuron statuses within a neural network.
- **Attributes**:
  - `_set_P`: Dictionary for Positive status neurons.
  - `_set_N`: Dictionary for Negative status neurons.
  - `_set_Z`: Dictionary for Zero status neurons.
  - `_set_U`: Dictionary for Unknown status neurons.
- **Methods**:
  - Getters and setters for each neuron status set.
  - `get_network(network)`: Initializes neuron statuses for each network layer.
  - `neuron_layer_is_in_set(set, NeuronS)`, `neuron_idx_is_in_layer(layer, NeuronS)`: Check methods for neuron's presence in sets or layers.
  - `is_in_set(set_type, NeuronS)`: Checks if a neuron is in a given set.
  - `add_to_set(set_type, NeuronS)`: Adds a neuron to a specific set.
  - `remove_from_set(set_type, NeuronS)`: Removes a neuron from a set.
  - `update_set(set_type, NeuronS)`, `move(from_set, to_set, NeuronS)`: Methods for updating neuron's set.
  - `display(set_type)`: Displays neurons in a specific set.

### NS (Inherits from NSBasic)
- **Purpose**: Extends `NSBasic` with additional functionalities.
- **Additional Attribute**:
  - `_SOI`: A set for additional, unspecified functionality.
- **Methods**:
  - Getter and setter for `_SOI`.
  - `init_SOI()`, `update_SOI()`: Placeholder methods for future implementation.

## Dependencies
- Relies on external modules like `Modules.utils`, `Modules.NNet`, and classes `NeuronStatus`, `NetworkStatus`.
