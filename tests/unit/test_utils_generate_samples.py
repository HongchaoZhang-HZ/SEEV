"""CI regressions for the optimized sample-scaling path."""

import warnings

import numpy as np
import pytest
import torch

from EEV.Modules.utils import generate_samples


def _legacy_generate_samples(domain, num_samples):
    domain = np.array(domain)
    samples = torch.rand((num_samples, len(domain)))
    return samples * (domain[:, 1] - domain[:, 0]) + domain[:, 0]


@pytest.mark.parametrize(
    "domain",
    [
        [(-1.0, 1.0), (0.0, 3.0)],
        [(0, 3), (-4, 7), (2, 2)],
        [(np.float32(-1), np.float32(2))],
    ],
)
def test_generate_samples_matches_legacy_bit_for_bit(domain):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="__array_wrap__ must accept context",
            category=DeprecationWarning,
        )
        torch.manual_seed(20260723)
        expected = _legacy_generate_samples(domain, 256)

    torch.manual_seed(20260723)
    actual = generate_samples(domain, 256)

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_generate_samples_scaling_is_warning_free():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        torch.manual_seed(20260723)
        generate_samples([(0, 3), (-4, 7)], 256)
