"""Deterministic tests for EEV sample-generation behavior.

Covers ``EEV.Modules.Function.generate_samples``: shape, dimensionality,
domain bounds, and seeded determinism.
"""

import torch

from EEV.Modules.Function import generate_samples


def test_shape_matches_count_and_dimension():
    samples = generate_samples([(-1, 1), (0, 2), (3, 4)], 7)
    assert tuple(samples.shape) == (7, 3)


def test_single_dimension_domain():
    samples = generate_samples([(-5, 5)], 4)
    assert tuple(samples.shape) == (4, 1)


def test_samples_stay_within_domain_bounds():
    domain = [(-1.0, 1.0), (2.0, 5.0)]
    torch.manual_seed(1234)
    samples = generate_samples(domain, 500)
    for dim, (lo, hi) in enumerate(domain):
        col = samples[:, dim]
        assert float(col.min()) >= lo
        assert float(col.max()) <= hi


def test_seeded_generation_is_deterministic():
    domain = [(-2.0, 2.0), (0.0, 1.0)]
    torch.manual_seed(7)
    first = generate_samples(domain, 16)
    torch.manual_seed(7)
    second = generate_samples(domain, 16)
    assert torch.equal(first, second)


def test_zero_samples_produces_empty_batch():
    samples = generate_samples([(-1, 1), (-1, 1)], 0)
    assert tuple(samples.shape) == (0, 2)


def test_degenerate_domain_collapses_to_bound():
    # A zero-width interval must yield exactly that constant on its dimension.
    samples = generate_samples([(3.0, 3.0), (-1.0, 1.0)], 10)
    expected = torch.full((10,), 3.0, dtype=samples.dtype)
    assert torch.equal(samples[:, 0], expected)
