"""Deterministic tests for EEV neural-network construction and evaluation.

Covers ``Modules.NNet.NeuralNetwork``: architecture-to-layer translation,
activation selection, device placement, forward evaluation with fixed weights,
and the linear layer-merge transform.
"""

import torch
import torch.nn as nn
import pytest

from Modules.NNet import NeuralNetwork


def _set_linear(layer, weight, bias):
    layer.weight.data = torch.tensor(weight, dtype=layer.weight.dtype)
    layer.bias.data = torch.tensor(bias, dtype=layer.bias.dtype)


def test_architecture_translates_to_layer_stack():
    net = NeuralNetwork([('relu', 2), ('relu', 3), ('linear', 1)])
    kinds = [type(layer).__name__ for layer in net.layers]
    assert kinds == ['Linear', 'ReLU', 'Linear', 'Identity']


def test_layer_feature_sizes():
    net = NeuralNetwork([('relu', 2), ('relu', 3), ('linear', 1)])
    assert net.layers[0].in_features == 2
    assert net.layers[0].out_features == 3
    assert net.layers[2].in_features == 3
    assert net.layers[2].out_features == 1


def test_activation_choices_map_to_modules():
    net = NeuralNetwork(
        [('linear', 1), ('relu', 1), ('sigmoid', 1), ('tanh', 1), ('linear', 1)]
    )
    activations = [type(l).__name__ for l in net.layers if not isinstance(l, nn.Linear)]
    assert activations == ['ReLU', 'Sigmoid', 'Tanh', 'Identity']


def test_unknown_activation_raises_value_error():
    with pytest.raises(ValueError):
        NeuralNetwork([('relu', 2), ('mystery', 3)])


def test_network_parameters_use_declared_device():
    net = NeuralNetwork([('relu', 2), ('linear', 1)])
    assert next(net.parameters()).device == net.device


def test_forward_linear_is_exact_affine_map():
    net = NeuralNetwork([('linear', 2), ('linear', 1)])
    _set_linear(net.layers[0], [[1.0, 2.0]], [0.5])
    out = net.forward(torch.tensor([3.0, 4.0]))
    # 1*3 + 2*4 + 0.5 = 11.5
    assert out.item() == pytest.approx(11.5)


def test_forward_relu_clamps_negative_preactivation():
    net = NeuralNetwork([('relu', 2), ('relu', 1)])
    _set_linear(net.layers[0], [[1.0, -1.0]], [0.0])
    assert net.forward(torch.tensor([1.0, 5.0])).item() == pytest.approx(0.0)
    assert net.forward(torch.tensor([5.0, 1.0])).item() == pytest.approx(4.0)


def test_forward_supports_batched_input():
    net = NeuralNetwork([('linear', 2), ('linear', 3)])
    out = net.forward(torch.zeros(4, 2))
    assert tuple(out.shape) == (4, 3)


def test_merge_last_n_layers_preserves_forward():
    net = NeuralNetwork([('linear', 6), ('linear', 4), ('linear', 3)])
    torch.manual_seed(0)
    _set_linear(
        net.layers[0],
        [[float((i * 6 + j) % 5 - 2) for j in range(6)] for i in range(4)],
        [0.1 * i for i in range(4)],
    )
    _set_linear(
        net.layers[2],
        [[float((i * 4 + j) % 3 - 1) for j in range(4)] for i in range(3)],
        [-0.2 * i for i in range(3)],
    )
    x = torch.tensor([0.5, -1.0, 2.0, 0.0, 1.5, -0.5])
    before = net.forward(x).detach().clone()
    net.merge_last_n_layers(2)
    after = net.forward(x)
    assert torch.allclose(before, after, atol=1e-5)
    # Two linear layers collapse to a single Linear followed by Identity.
    assert [type(l).__name__ for l in net.layers] == ['Linear', 'Identity']


def test_merge_last_n_layers_rejects_too_few_layers():
    net = NeuralNetwork([('linear', 6), ('linear', 3)])
    with pytest.raises(ValueError):
        net.merge_last_n_layers(1)
