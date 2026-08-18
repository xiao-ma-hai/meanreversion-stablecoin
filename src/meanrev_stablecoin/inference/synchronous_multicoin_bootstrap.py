from __future__ import annotations

import numpy as np


def synchronous_day_resample(
    days_by_asset: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Resample the same common UTC days and return indices for every asset.

    Days absent from any asset are excluded before resampling.  The returned
    per-asset indices preserve all within-day observations and therefore retain
    intraday serial dependence while aligning market-wide shocks across assets.
    """

    if not days_by_asset:
        raise ValueError("at least one asset is required")
    common: set[object] | None = None
    for days in days_by_asset.values():
        values = set(np.asarray(days).tolist())
        common = values if common is None else common.intersection(values)
    common_days = np.asarray(sorted(common or ()))
    if len(common_days) < 2:
        raise ValueError("at least two common UTC days are required")
    sampled_days = common_days[rng.integers(0, len(common_days), size=len(common_days))]
    indices: dict[str, np.ndarray] = {}
    for asset, days in days_by_asset.items():
        array = np.asarray(days)
        by_day = {day: np.flatnonzero(array == day) for day in common_days}
        indices[asset] = np.concatenate([by_day[day] for day in sampled_days])
    return sampled_days, indices
