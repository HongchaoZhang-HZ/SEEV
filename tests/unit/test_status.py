"""Deterministic tests for EEV neuron/network status behaviour.

Covers ``Scripts.Status.NeuronStatus`` and ``Scripts.Status.NetworkStatus``:
status encoding, tolerance-based sign classification, and forward-propagation
of activation status through a fixed-weight network.
"""

import torch
import pytest

import Scripts.PARA as PARA
from Modules.NNet import NeuralNetwork
from Scripts.Status import NeuronStatus, NetworkStatus


def _set_linear(layer, weight, bias):
    layer.weight.data = torch.tensor(weight, dtype=layer.weight.dtype)
    layer.bias.data = torch.tensor(bias, dtype=layer.bias.dtype)


def test_neuron_status_stores_identity_and_status():
    ns = NeuronStatus(1, 2, 0)
    assert ns.layer == 1
    assert ns.neuron == 2
    assert ns.status == 0


def test_neuron_status_defaults_to_unknown_when_none():
    ns = NeuronStatus(0, 0, None)
    assert ns.status == -2


def test_neuron_status_get_id():
    assert NeuronStatus(3, 4, 1).get_id() == [3, 4]


def test_neuron_status_get_and_set_status():
    ns = NeuronStatus(0, 0, -2)
    assert ns.get_status() == -2
    ns.set_status(1)
    assert ns.get_status() == 1


def test_neuron_status_tolerance_from_para():
    assert NeuronStatus(0, 0, -2).tol == PARA.zero_tol


def test_set_status_from_value_positive():
    ns = NeuronStatus(0, 0, -2)
    ns.set_status_from_value(0.5)
    assert ns.status == 1


def test_set_status_from_value_negative():
    ns = NeuronStatus(0, 0, -2)
    ns.set_status_from_value(-0.5)
    assert ns.status == -1


def test_set_status_from_value_zero_within_tolerance():
    ns = NeuronStatus(0, 0, -2)
    ns.set_status_from_value(0.0)
    assert ns.status == 0


def test_network_status_initialises_empty():
    net = NeuralNetwork([('relu', 2), ('linear', 1)])
    status = NetworkStatus(net)
    assert status.neuron_inputs == {}
    assert status.network_status == {}
    assert status.network_status_values == {}


def test_set_layer_status_from_value_classifies_signs():
    net = NeuralNetwork([('relu', 2), ('linear', 1)])
    status = NetworkStatus(net)
    layer = status.set_layer_status_from_value(0, [0.5, -0.5, 0.0])
    assert [n.status for n in layer] == [1, -1, 0]


def test_set_layer_status_from_value_uses_halved_layer_index():
    net = NeuralNetwork([('relu', 2), ('linear', 1)])
    status = NetworkStatus(net)
    layer = status.set_layer_status_from_value(4, [0.1, 0.2])
    assert all(n.layer == 2 for n in layer)


def test_get_netstatus_from_input_single_output():
    net = NeuralNetwork([('linear', 2), ('linear', 1)])
    _set_linear(net.layers[0], [[1.0, -1.0]], [0.0])
    status = NetworkStatus(net)
    status.get_netstatus_from_input(torch.tensor([5.0, 1.0]))
    assert status.network_status_values[0] == [1]
    assert status.neuron_inputs[0] == pytest.approx([4.0])


def test_get_netstatus_from_input_multi_output_signs():
    net = NeuralNetwork([('linear', 2), ('linear', 2)])
    _set_linear(net.layers[0], [[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
    status = NetworkStatus(net)
    status.get_netstatus_from_input(torch.tensor([3.0, 5.0]))
    assert status.network_status_values[0] == [1, -1]


def test_get_netstatus_values_match_status_objects():
    net = NeuralNetwork([('linear', 2), ('linear', 2)])
    _set_linear(net.layers[0], [[1.0, 0.0], [0.0, -1.0]], [0.0, 0.0])
    status = NetworkStatus(net)
    status.get_netstatus_from_input(torch.tensor([2.0, 2.0]))
    for layer_idx, objs in status.network_status.items():
        assert [o.status for o in objs] == status.network_status_values[layer_idx]
