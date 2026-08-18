from __future__ import annotations

import numpy as np

from meanrev_stablecoin.inference.synchronous_multicoin_bootstrap import synchronous_day_resample


def test_synchronous_bootstrap_uses_same_day_multiplicities() -> None:
    days = {
        "USDT": np.array([1, 1, 2, 2, 3, 3]),
        "USDC": np.array([1, 2, 2, 3]),
        "DAI": np.array([1, 1, 1, 2, 3]),
    }
    sampled, indices = synchronous_day_resample(days, np.random.default_rng(5))
    expected_counts = {day: int(np.sum(sampled == day)) for day in np.unique(sampled)}
    for asset, selected in indices.items():
        selected_days = days[asset][selected]
        for day, multiplicity in expected_counts.items():
            original_count = int(np.sum(days[asset] == day))
            assert int(np.sum(selected_days == day)) == multiplicity * original_count
