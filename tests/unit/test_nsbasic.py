"""Deterministic tests for EEV activation set-management behaviour.

Covers ``Scripts.NSBasic.NSBasic`` (the P/N/Z/U set bookkeeping) and
``Scripts.NSBasic.NS`` (network-derived unknown-set initialisation).
"""

from Modules.NNet import NeuralNetwork
from Scripts.NSBasic import NSBasic, NS
from Scripts.Status import NeuronStatus


def test_sets_start_empty():
    nsb = NSBasic()
    assert nsb.set_P == {}
    assert nsb.set_N == {}
    assert nsb.set_Z == {}
    assert nsb.set_U == {}


def test_set_property_setter_roundtrip():
    nsb = NSBasic()
    nsb.set_P = {0: {0: 'x'}}
    assert nsb.set_P == {0: {0: 'x'}}


def test_add_to_set_positive_assigns_status_one():
    nsb = NSBasic()
    nsb.add_to_set('P', NeuronStatus(0, 0, -2))
    assert nsb.set_P[0][0].status == 1


def test_add_to_set_assigns_status_per_type():
    expected = {'N': -1, 'Z': 0, 'U': -2}
    for set_type, status in expected.items():
        nsb = NSBasic()
        nsb.add_to_set(set_type, NeuronStatus(0, 0, 7))
        stored = getattr(nsb, f'set_{set_type}')[0][0]
        assert stored.status == status


def test_is_in_set_reflects_membership():
    nsb = NSBasic()
    neuron = NeuronStatus(1, 3, -2)
    assert nsb.is_in_set('U', neuron) is False
    nsb.add_to_set('U', neuron)
    assert nsb.is_in_set('U', neuron) is True


def test_neuron_layer_is_in_set():
    nsb = NSBasic()
    neuron = NeuronStatus(0, 0, -2)
    nsb.set_U = {0: {0: neuron}}
    assert nsb.neuron_layer_is_in_set(nsb.set_U, neuron) is True
    assert nsb.neuron_layer_is_in_set(nsb.set_U, NeuronStatus(5, 0, -2)) is False


def test_neuron_idx_is_in_layer():
    nsb = NSBasic()
    neuron = NeuronStatus(0, 2, -2)
    nsb.set_U = {0: {2: neuron}}
    assert nsb.neuron_idx_is_in_layer(nsb.set_U[0], neuron) is True
    assert nsb.neuron_idx_is_in_layer(nsb.set_U[0], NeuronStatus(0, 9, -2)) is False


def test_remove_from_set():
    nsb = NSBasic()
    neuron = NeuronStatus(0, 0, -2)
    nsb.add_to_set('U', neuron)
    nsb.remove_from_set('U', neuron)
    assert nsb.set_U[0] == {}


def test_move_transfers_membership_and_status():
    nsb = NSBasic()
    neuron = NeuronStatus(0, 0, -2)
    nsb.add_to_set('U', neuron)
    nsb.move('U', 'P', neuron)
    assert nsb.set_U[0] == {}
    assert nsb.set_P[0][0].status == 1


def test_update_set_refreshes_status():
    nsb = NSBasic()
    nsb.add_to_set('U', NeuronStatus(0, 0, -2))
    nsb.update_set('U', NeuronStatus(0, 0, -1))
    # Re-adding to U resets the stored status back to U's canonical value.
    assert nsb.set_U[0][0].status == -2


def test_add_to_set_is_idempotent_for_same_neuron():
    nsb = NSBasic()
    neuron = NeuronStatus(0, 0, -2)
    nsb.add_to_set('U', neuron)
    nsb.add_to_set('U', neuron)
    assert len(nsb.set_U[0]) == 1


def test_ns_initialises_empty_sets():
    ns = NS()
    assert ns.set_P == {} and ns.set_N == {}
    assert ns.set_Z == {} and ns.set_U == {}
    assert ns.SOI == {}


def test_ns_get_network_builds_unknown_set_shape():
    ns = NS()
    ns.get_network(NeuralNetwork([('relu', 2), ('relu', 32), ('linear', 1)]))
    assert sorted(ns.set_U.keys()) == [0, 1]
    assert len(ns.set_U[0]) == 32
    assert len(ns.set_U[1]) == 1


def test_ns_get_network_marks_neurons_unknown():
    ns = NS()
    ns.get_network(NeuralNetwork([('relu', 2), ('relu', 32), ('linear', 1)]))
    neuron = ns.set_U[0][5]
    assert neuron.get_id() == [0, 5]
    assert neuron.status == -2
