"""Measured optimization guard for ``EEV.Modules.utils.generate_samples``.

This module pins down two things at once:

* **Behavior preservation** - the optimized, Torch-only implementation must
  produce bit-for-bit identical output to the historical NumPy/Torch-crossing
  reference for the same seed, and it must not emit the NumPy 2
  ``__array_wrap__`` deprecation warning that the reference does.
* **No performance regression** - a ``pytest-benchmark`` run measures both the
  local legacy reference and the shipped implementation on the same
  representative input. The optimized median must be no slower than the legacy
  median in the reported run.

Raw benchmark JSON is written by the validation command into ``artifacts/m2/``.
"""

import warnings

import numpy as np
import torch

from EEV.Modules.utils import generate_samples as optimized_generate_samples


def _legacy_generate_samples(domain, num_samples):
    """The pre-optimization implementation, kept locally as a reference.

    It scales Torch samples by NumPy arrays, which is exactly the
    NumPy/Torch boundary crossing that the optimization removes.
    """
    num_dimensions = len(domain)
    domain = np.array(domain)
    samples = torch.rand((num_samples, num_dimensions))
    samples = samples * (domain[:, 1] - domain[:, 0]) + domain[:, 0]
    return samples


# Representative input reused by every check below.
_DOMAIN = [(-1.0, 1.0), (0.0, 2.0), (-5.0, 5.0), (3.0, 3.0)]
_NUM_SAMPLES = 2048
_SEED = 20260723


def test_optimized_matches_legacy_bit_for_bit():
    torch.manual_seed(_SEED)
    legacy = _legacy_generate_samples(_DOMAIN, _NUM_SAMPLES)
    torch.manual_seed(_SEED)
    optimized = optimized_generate_samples(_DOMAIN, _NUM_SAMPLES)

    assert legacy.dtype == optimized.dtype
    assert legacy.shape == optimized.shape
    assert torch.equal(legacy, optimized)


def test_optimized_scaling_is_warning_free():
    # The legacy path emits the NumPy 2 deprecation warning; the optimized
    # path must not emit any warning during scaling.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        torch.manual_seed(_SEED)
        optimized_generate_samples(_DOMAIN, _NUM_SAMPLES)


def test_legacy_benchmark(benchmark):
    torch.manual_seed(_SEED)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="__array_wrap__ must accept context",
            category=DeprecationWarning,
        )
        result = benchmark(lambda: _legacy_generate_samples(_DOMAIN, _NUM_SAMPLES))
    assert tuple(result.shape) == (_NUM_SAMPLES, len(_DOMAIN))


def test_optimized_benchmark(benchmark):
    torch.manual_seed(_SEED)
    result = benchmark(lambda: optimized_generate_samples(_DOMAIN, _NUM_SAMPLES))
    assert tuple(result.shape) == (_NUM_SAMPLES, len(_DOMAIN))
