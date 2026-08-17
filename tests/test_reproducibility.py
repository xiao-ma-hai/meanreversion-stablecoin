import numpy as np

from meanrev_stablecoin.estimation.bootstrap import block_bootstrap_indices


def test_fixed_seed_reproduces_bootstrap_indices():
    first = block_bootstrap_indices(1000, 50, np.random.default_rng(20260806))
    second = block_bootstrap_indices(1000, 50, np.random.default_rng(20260806))
    assert np.array_equal(first, second)

