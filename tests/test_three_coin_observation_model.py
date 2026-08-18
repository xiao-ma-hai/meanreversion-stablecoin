from __future__ import annotations

import numpy as np

from meanrev_stablecoin.models.mou_interval_likelihood import mixed_ou_interval_logprob_tau


def test_same_interval_architecture_accepts_asset_specific_ticks() -> None:
    previous = np.array([1.0, 1.0, 0.9999, 1.0001])
    following = np.array([1.0, 1.0001, 1.0, 1.0001])
    delta = np.full(4, 1 / 288)
    for tick in (1e-4, 1e-5):
        values = mixed_ou_interval_logprob_tau(
            following, previous, tick, delta, 0.0, 0.001, 1.5, 2.0
        )
        assert values.shape == following.shape
        assert np.isfinite(values).all()
