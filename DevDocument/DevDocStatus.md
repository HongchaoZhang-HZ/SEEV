# Status.py

`Status.py` is a Python script that defines a class `NetworkStatus` for managing the status of a neural network.

# Python File Summary

## Overview
This Python script is designed to track and display the status of neurons within a neural network. It defines two main classes, `NeuronStatus` and `NetworkStatus`, and includes a main section for creating a neural network and processing a random input.

## Classes

### NeuronStatus
- **Purpose**: Represents the status of a single neuron in the network.
- **Attributes**:
  - `layer`: The layer index of the neuron.
  - `neuron`: The neuron index within its layer.
  - `status`: Indicates the neuron's state (-2 for unknown, -1 for negative, 0 for zero, 1 for positive).
  - `tol`: Tolerance level used to determine the status based on a value.
- **Methods**:
  - `get_id()`: Returns the layer and neuron indices.
  - `get_status()`, `set_status()`: Getters and setters for the neuron's status.
  - `set_status_from_value(value)`: Sets the neuron's status based on a given value.
  - `display()`: Prints the neuron's layer, index, and status.

### NetworkStatus
- **Purpose**: Manages the status of the entire neural network.
- **Attributes**:
  - `network`: The neural network object.
  - `neuron_inputs`: Dictionary storing inputs to each neuron.
  - `network_status`: Stores the status of each neuron.
  - `network_status_values`: Stores the status values of each neuron.
- **Methods**:
  - `set_layer_status(layer_idx, layer_status)`: Sets the status of neurons in a specified layer.
  - `set_layer_status_from_value(layer_idx, input_value)`: Determines and sets neuron statuses based on input values.
  - `get_neuron_inputs()`: Returns the neuron inputs.
  - `display_layer_status(layer_status)`: Displays the status of a specific layer.
  - `get_netstatus_from_input(input_value)`: Processes an input through the network, updating neuron statuses.
  - `display_network_status_value()`: Displays the status values of the network.

## Main Section
- **Functionality**:
  - Creates a neural network model with a predefined architecture.
  - Generates a random input for the model.
  - Initializes a `NetworkStatus` object and processes the input through the network.

## Remarks
- The script is intended for debugging or analyzing neural network behavior.
- Dependencies include the `Modules.utils` and `Modules.NNet` modules, and the PyTorch library.
- Note: Logging functionality (writing to a log file) is not implemented in the provided code.
